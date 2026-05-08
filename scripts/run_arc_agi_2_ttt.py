import os
import json
import time
import torch
import torch.nn.functional as F
from typing import Dict, List, Any

# Internal RADS Modules
from models.diffusion_prior import MaskedDiffusionPrior, release_cuda_fragments
from models.trm_verifier import get_compiled_trm
from models.sequence_packing import GridSequencePacker
from data.transforms import apply_paired_color_permutation, apply_random_symmetry_group
from core.constants import SEP_TOKEN_ID
from core.grid_codec import dummy_prediction_from_input

# Configuration Constants
TTT_STEPS = 150
DIFFUSION_STEPS_INFERENCE = 10
MAX_RUNTIME_SEC = (
    11.5 * 3600
)  # 11.5 hours to safely stay under the 12-hour Kaggle limit
CANDIDATES_PER_TEST = 16  # Number of diffusion hypotheses to generate per test input


def augment_demonstrations(
    train_pairs: List[Dict[str, List[List[int]]]], num_samples: int = 32
) -> List[Dict[str, torch.Tensor]]:
    """
    Applies stateless, CPU-bound geometric and semantic augmentations to the
    task's demonstration pairs to build a robust Test-Time Training (TTT) dataset.
    """
    augmented_data = []

    for _ in range(num_samples):
        # Uniformly sample one of the demonstration pairs
        pair_idx = int(torch.randint(0, len(train_pairs), (1,)).item())
        base_pair = train_pairs[pair_idx]

        inp_grid = torch.tensor(base_pair["input"], dtype=torch.long).numpy()
        out_grid = torch.tensor(base_pair["output"], dtype=torch.long).numpy()

        # Apply stochastic augmentations
        if torch.rand(1).item() < 0.5:
            inp_grid, out_grid = apply_paired_color_permutation(
                inp_grid, out_grid, num_colors=10
            )

        if torch.rand(1).item() < 0.5:
            inp_grid, out_grid = apply_random_symmetry_group(inp_grid, out_grid)

        augmented_data.append(
            {
                "input": torch.from_numpy(inp_grid.copy()),
                "output": torch.from_numpy(out_grid.copy()),
            }
        )

    return augmented_data


def execute_ttt_loop(
    model: MaskedDiffusionPrior,
    train_pairs: List[Dict[str, List[List[int]]]],
    optimizer: torch.optim.Optimizer,
):
    """
    Executes the Test-Time Training loop, updating the LoRA adapter weights
    on the augmented demonstration pairs to adapt to the novel rule structure.
    """
    model.train()

    for step in range(TTT_STEPS):
        optimizer.zero_grad()

        # 1. Generate a mini-batch of augmented demonstrations
        batch_data = augment_demonstrations(train_pairs, num_samples=4)

        # 2. Pack the variable-sized grids into a contiguous 1D NestedTensor format
        # For MLM, we concatenate the input and output grids into a single sequence
        packed_sequences = []
        for pair in batch_data:
            # Flatten and concatenate: [INPUT_TOKENS, <SEP>, OUTPUT_TOKENS]
            seq = torch.cat(
                [
                    pair["input"].flatten(),
                    torch.tensor([SEP_TOKEN_ID]),
                    pair["output"].flatten(),
                ]
            )
            packed_sequences.append(seq)

        pack_info = GridSequencePacker.pack_grids(packed_sequences, device=model.device)
        packed_tensor = pack_info["packed_sequence"]

        # 3. Apply random masking for the diffusion objective
        mask_prob = (
            torch.rand(1).item() * 0.8 + 0.1
        )  # Mask between 10% and 90% of tokens
        mask_indices = (
            torch.rand(packed_tensor.shape, device=packed_tensor.device) < mask_prob
        )
        if not bool(mask_indices.any().item()):
            mask_indices[0] = True

        # Create soft-token distributions
        soft_tokens = F.one_hot(packed_tensor, num_classes=model.vocab_size).float()
        soft_tokens[mask_indices] = 0.0
        soft_tokens[mask_indices, model.mask_token_id] = 1.0

        # 4. Forward pass (Continuous Token Algebra)
        refined_tokens = model.continuous_denoise_step(
            packed_soft_tokens=soft_tokens,
            cu_seq_lens=pack_info["cu_seq_lens"],
            max_seq_len=pack_info["max_seq_len"],
            row_coords=pack_info["row_coords"],
            col_coords=pack_info["col_coords"],
        )

        # 5. Compute Cross-Entropy Loss strictly on the masked tokens
        loss = F.nll_loss(
            refined_tokens[mask_indices].clamp_min(1e-8).log(),
            packed_tensor[mask_indices],
        )

        # 6. Backpropagate and update LoRA adapter
        loss.backward()
        optimizer.step()


@torch.inference_mode()
def generate_and_verify(
    model: MaskedDiffusionPrior,
    trm_verifier: torch.nn.Module,
    test_input: List[List[int]],
) -> List[List[List[int]]]:
    """
    Generates multiple candidate outputs using the Diffusion Prior, scores them
    using the TRM Verifier, and returns the top 2 attempts.
    """
    model.eval()
    device = model.device
    test_tensor = torch.tensor(test_input, dtype=torch.long, device=device).flatten()

    candidates = []

    # Generate multiple hypotheses
    for _ in range(CANDIDATES_PER_TEST):
        # We assume the model predicts the output dimensions as part of its generative process
        # For simplicity in this script, we assume the output shape matches the input shape
        h, w = len(test_input), len(test_input[0])
        dummy_output = torch.full(
            (h * w,), model.mask_token_id, dtype=torch.long, device=device
        )

        seq = torch.cat(
            [test_tensor, torch.tensor([SEP_TOKEN_ID], device=device), dummy_output]
        )
        pack_info = GridSequencePacker.pack_grids([seq], device=device)

        # Diffuse
        hard_tokens = model.generate_hypothesis(
            packed_context=pack_info["packed_sequence"],
            cu_seq_lens=pack_info["cu_seq_lens"],
            max_seq_len=pack_info["max_seq_len"],
            row_coords=pack_info["row_coords"],
            col_coords=pack_info["col_coords"],
            num_diffusion_steps=DIFFUSION_STEPS_INFERENCE,
        )

        # Extract the output segment and reshape
        output_tokens = hard_tokens[len(test_tensor) + 1 :]
        candidates.append(output_tokens.view(h, w).cpu().numpy().tolist())

    if not candidates:
        fallback = dummy_prediction_from_input(test_input)
        return [fallback, fallback]

    # Verify candidates with TRM. This runner uses a deterministic lightweight
    # feature encoder so the compiled TRM path is exercised even when trained
    # encoder weights are not available in the local workspace.
    latents = _encode_candidates_for_trm(
        candidates, device=next(trm_verifier.parameters()).device
    )
    trace = trm_verifier(latents)
    logits = trace.logits.squeeze(-1)
    scores = []
    for idx, cand in enumerate(candidates):
        converged_bonus = 1.0 if bool(trace.converged[idx].item()) else 0.0
        score = (
            converged_bonus,
            -float(trace.final_delta[idx].item()),
            float(logits[idx].item()),
        )
        scores.append((score, cand))
    scores.sort(key=lambda item: item[0], reverse=True)

    unique_candidates = []
    for _, cand in scores:
        if cand not in unique_candidates:
            unique_candidates.append(cand)
        if len(unique_candidates) == 2:
            break

    # Pad to exactly 2 attempts as required by ARC-AGI-2 rules
    while len(unique_candidates) < 2:
        unique_candidates.append(unique_candidates[0])

    return unique_candidates


def _encode_candidates_for_trm(
    candidates: List[List[List[int]]], device: torch.device, embed_dim: int = 512
) -> torch.Tensor:
    """Encode candidate grids into fixed-size deterministic TRM feature vectors."""

    latents = torch.zeros(
        (len(candidates), embed_dim), dtype=torch.float32, device=device
    )
    for i, cand in enumerate(candidates):
        tensor = torch.tensor(cand, dtype=torch.float32, device=device)
        flat = tensor.flatten()
        hist = torch.bincount(flat.to(torch.long).clamp(0, 15), minlength=16).float()
        hist = hist / hist.sum().clamp_min(1.0)
        features = torch.cat(
            [
                hist,
                torch.tensor(
                    [
                        tensor.shape[0],
                        tensor.shape[1],
                        flat.mean(),
                        flat.std(unbiased=False),
                    ],
                    device=device,
                ),
            ]
        )
        repeats = (embed_dim + features.numel() - 1) // features.numel()
        latents[i] = features.repeat(repeats)[:embed_dim]
    return latents


def _fallback_attempts_for_task(
    task_data: Dict[str, Any],
) -> List[Dict[str, List[List[int]]]]:
    """Return valid two-attempt fallback predictions for every test item in a task."""

    attempts = []
    for test_pair in task_data.get("test", []):
        fallback = dummy_prediction_from_input(test_pair["input"])
        attempts.append({"attempt_1": fallback, "attempt_2": fallback})
    return attempts


def main():
    print("=== RADS ARC-AGI-2 Test-Time Training Engine ===")

    start_time = time.time()

    # 1. Load the task data
    # During submission, Kaggle swaps this file with the actual hidden test set
    data_path = "/kaggle/input/arc-prize-2026/arc-agi_test-challenges.json"
    if not os.path.exists(data_path):
        # Fallback for local development
        data_path = "data/arc-agi_evaluation-challenges.json"

    with open(data_path, "r") as f:
        tasks = json.load(f)

    print(f"Loaded {len(tasks)} tasks for evaluation.")

    # 2. Initialize Neural Core
    print("Loading 8B Masked Diffusion Prior (NF4 QLoRA)...")
    # Base model ID would be the path to the offline weights in the Kaggle environment
    base_model_path = os.environ.get("RADS_BASE_MODEL_DIR")
    use_tiny = (
        os.environ.get("RADS_USE_TINY_BACKBONE", "0") == "1" or base_model_path is None
    )
    if use_tiny:
        print("WARNING: RADS_BASE_MODEL_DIR is not set; using tiny offline backbone.")
    model = MaskedDiffusionPrior(
        base_model_id=base_model_path, use_tiny_backbone=use_tiny
    )

    print("Compiling 7M TRM Verifier...")
    trm_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trm_verifier = get_compiled_trm(device=trm_device)

    # Initialize FP16 LoRA Optimizer
    # We only optimize the parameters that require gradients (the LoRA adapters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4
    )

    # 3. Process each task
    submission_dict = {}

    for i, (task_id, task_data) in enumerate(tasks.items()):
        # Hard timeout check
        if time.time() - start_time > MAX_RUNTIME_SEC:
            print("WARNING: Approaching 12-hour limit. Truncating remaining tasks.")
            for remaining_task_id, remaining_task_data in list(tasks.items())[i:]:
                submission_dict[remaining_task_id] = _fallback_attempts_for_task(
                    remaining_task_data
                )
            break

        print(f"Processing Task {i + 1}/{len(tasks)}: {task_id}")

        try:
            # Test-Time Training Phase
            train_pairs = task_data["train"]
            execute_ttt_loop(model, train_pairs, optimizer)
            release_cuda_fragments()

            # Inference & Verification Phase
            submission_dict[task_id] = []
            for test_pair in task_data["test"]:
                predictions = generate_and_verify(
                    model, trm_verifier, test_pair["input"]
                )

                # Format requires exactly two attempts per test input
                submission_dict[task_id].append(
                    {"attempt_1": predictions[0], "attempt_2": predictions[1]}
                )
        except Exception as exc:
            print(
                f"ERROR: Task {task_id} failed with {type(exc).__name__}: {exc}. Emitting fallback attempts."
            )
            release_cuda_fragments()
            submission_dict[task_id] = _fallback_attempts_for_task(task_data)

    # 4. Save Submission
    with open("submission.json", "w") as f:
        json.dump(submission_dict, f)

    print(
        f"Submission saved. Total runtime: {(time.time() - start_time) / 3600:.2f} hours."
    )


if __name__ == "__main__":
    main()
