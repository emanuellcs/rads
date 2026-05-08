import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Internal RADS Modules
from models.diffusion_prior import MaskedDiffusionPrior
from models.trm_verifier import TinyRecursiveVerifier
from models.sequence_packing import GridSequencePacker
from data.dataset import create_arc_dataloader
from core.constants import SEP_TOKEN_ID, VOCAB_SIZE

# ==========================================
# Dummy RE-ARC Registry for Demonstration
# ==========================================
# In the real pipeline, this imports from data.re_arc_generators
import numpy as np


def dummy_fill_generator():
    """A trivial placeholder for a RE-ARC procedural generator."""
    inp = np.random.randint(0, 10, (10, 10))
    out = inp.copy()
    out[out == 0] = 1
    return inp, out


GENERATOR_REGISTRY = {
    "flood_fill": dummy_fill_generator,
    "mirror_x": dummy_fill_generator,
    # ... ~1,000 other concepts ...
}


# ==========================================
# TRM Sequence Pooling Encoder
# ==========================================
class TRMEncoder(nn.Module):
    """
    Maps the variable-length packed sequence into the fixed-size
    latent dimension (d_z = 512) required by the TRM verifier.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(
        self, packed_tokens: torch.Tensor, cu_seq_lens: torch.Tensor
    ) -> torch.Tensor:
        """Mean-pools the embedded tokens sequence-by-sequence."""
        embeds = self.embedding(packed_tokens)  # [total_tokens, embed_dim]
        embeds = self.proj(embeds)

        batch_size = len(cu_seq_lens) - 1
        pooled = torch.zeros((batch_size, embeds.shape[-1]), device=embeds.device)

        # Pool each sequence independently
        for i in range(batch_size):
            start, end = cu_seq_lens[i], cu_seq_lens[i + 1]
            pooled[i] = embeds[start:end].mean(dim=0)

        return pooled


# ==========================================
# Main Pre-Training Loop
# ==========================================


def main():
    print("=== RADS Phase 1: Offline Pre-Training Engine ===")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Initialize the CoW-Leak-Free Procedural DataLoader
    print("Initializing stateless DataLoader...")
    dataloader = create_arc_dataloader(
        generator_registry=GENERATOR_REGISTRY, batch_size=32, num_workers=4
    )

    # 2. Initialize Models
    print("Loading 8B Masked Diffusion Prior (NF4 QLoRA)...")
    # Using Llama-3-8B as the base prior
    base_model_path = os.environ.get("RADS_BASE_MODEL_DIR")
    use_tiny = (
        os.environ.get("RADS_USE_TINY_BACKBONE", "0") == "1" or base_model_path is None
    )
    if use_tiny:
        print("WARNING: RADS_BASE_MODEL_DIR is not set; using tiny offline backbone.")
    diffusion_model = MaskedDiffusionPrior(
        base_model_id=base_model_path, use_tiny_backbone=use_tiny
    )

    print("Initializing 7M TRM Verifier & Encoder...")
    trm_encoder = TRMEncoder(embed_dim=512).to(device)
    trm_verifier = TinyRecursiveVerifier(embed_dim=512).to(device)

    # Note: We do NOT use `torch.compile` during Phase 1 training.
    # Graph capture is strictly for ultra-fast frozen-weight inference.

    # 3. Setup Optimizers
    # Only optimize the LoRA adapter weights, NOT the 4-bit base model
    lora_params = [p for p in diffusion_model.parameters() if p.requires_grad]

    opt_diffusion = AdamW(lora_params, lr=1e-4, weight_decay=0.01)
    opt_trm = AdamW(
        list(trm_encoder.parameters()) + list(trm_verifier.parameters()), lr=3e-4
    )

    scheduler_diff = CosineAnnealingLR(opt_diffusion, T_max=100000)
    scheduler_trm = CosineAnnealingLR(opt_trm, T_max=100000)

    # 4. Training Loop
    print("Beginning Phase 1 Training...")
    diffusion_model.train()
    trm_verifier.train()
    trm_encoder.train()

    step = 0
    start_time = time.time()

    for batch in dataloader:
        step += 1

        # --- PREPARE DATA ---
        input_grids = [sample["input_grid"] for sample in batch]
        output_grids = [sample["output_grid"] for sample in batch]

        packed_sequences = []
        for inp, out in zip(input_grids, output_grids):
            # Format: [INPUT_TOKENS, <SEP>, OUTPUT_TOKENS]
            seq = torch.cat(
                [inp.flatten(), torch.tensor([SEP_TOKEN_ID]), out.flatten()]
            ).to(device)
            packed_sequences.append(seq)

        pack_info = GridSequencePacker.pack_grids(packed_sequences, device=device)
        packed_tensor = pack_info["packed_sequence"]
        cu_seq_lens = pack_info["cu_seq_lens"]

        # ==========================================
        # OBJECTIVE 1: MASKED DIFFUSION TRAINING
        # ==========================================
        opt_diffusion.zero_grad()

        # Dynamic Masking (mask 15% to 85% of target tokens)
        mask_prob = torch.rand(1).item() * 0.70 + 0.15
        mask_indices = torch.rand(packed_tensor.shape, device=device) < mask_prob
        if not bool(mask_indices.any().item()):
            mask_indices[0] = True

        soft_tokens = F.one_hot(
            packed_tensor, num_classes=diffusion_model.vocab_size
        ).float()
        soft_tokens[mask_indices] = 0.0
        soft_tokens[mask_indices, diffusion_model.mask_token_id] = 1.0

        refined_tokens = diffusion_model.continuous_denoise_step(
            packed_soft_tokens=soft_tokens,
            cu_seq_lens=cu_seq_lens,
            max_seq_len=pack_info["max_seq_len"],
            row_coords=pack_info["row_coords"],
            col_coords=pack_info["col_coords"],
        )

        loss_diffusion = F.nll_loss(
            refined_tokens[mask_indices].clamp_min(1e-8).log(),
            packed_tensor[mask_indices],
        )
        loss_diffusion.backward()

        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt_diffusion.step()
        scheduler_diff.step()

        # ==========================================
        # OBJECTIVE 2: TRM THERMODYNAMIC VERIFICATION
        # ==========================================
        opt_trm.zero_grad()

        # 1. Positive Samples (Ground Truth)
        z_pos = trm_encoder(packed_tensor, cu_seq_lens)
        trace_pos = trm_verifier(z_pos, max_steps=32)

        # L2 distance between step K and K-1 (Should approach 0 for correct hypotheses)
        pos_dist = trace_pos.final_delta.mean()

        # 2. Negative Samples (Adversarially Corrupted Grids)
        # We simulate "incorrect hypotheses" by shifting the token values randomly
        corrupted_tensor = packed_tensor.clone()
        corruption_mask = torch.rand(corrupted_tensor.shape, device=device) < 0.2
        corrupted_tensor[corruption_mask] = torch.randint(
            0, 10, (corruption_mask.sum(),), device=device
        )

        z_neg = trm_encoder(corrupted_tensor, cu_seq_lens)
        trace_neg = trm_verifier(z_neg, max_steps=32)

        # L2 distance for corrupted (Should diverge, so distance > epsilon)
        neg_dist = trace_neg.final_delta.mean()

        # Contrastive Contraction Margin Loss
        # We want pos_dist to be 0, and neg_dist to be at least the margin (e.g., 5.0)
        margin = 5.0
        loss_trm = pos_dist + torch.relu(margin - neg_dist)

        loss_trm.backward()
        torch.nn.utils.clip_grad_norm_(
            list(trm_encoder.parameters()) + list(trm_verifier.parameters()), 1.0
        )
        opt_trm.step()
        scheduler_trm.step()

        # --- LOGGING & CHECKPOINTING ---
        if step % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"Step {step:05d} | Diff Loss: {loss_diffusion.item():.4f} | TRM Pos Dist: {pos_dist.item():.4f} | TRM Neg Dist: {neg_dist.item():.4f} | Time: {elapsed:.1f}s"
            )

        if step % 5000 == 0:
            print(f"Saving Checkpoint at Step {step}...")
            os.makedirs("checkpoints", exist_ok=True)

            # Save LoRA Adapter
            diffusion_model.model.save_pretrained(f"checkpoints/rads_lora_step_{step}")

            # Save TRM Verifier & Encoder
            torch.save(
                {
                    "encoder_state_dict": trm_encoder.state_dict(),
                    "verifier_state_dict": trm_verifier.state_dict(),
                },
                f"checkpoints/rads_trm_step_{step}.pt",
            )


if __name__ == "__main__":
    main()
