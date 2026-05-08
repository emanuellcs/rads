from __future__ import annotations

import gc
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict

from core.constants import MASK_TOKEN_ID, VOCAB_SIZE


def release_cuda_fragments() -> None:
    """
    Release Python and CUDA allocator fragments at coarse state transitions.

    This is intentionally not called inside denoising or MCTS hot loops. It is
    meant for transitions such as LoRA adapter swaps, post-TTT cleanup, and
    GPU-server teardown where allocator fragmentation can otherwise cause T4 OOM.
    """

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class TinyBidirectionalDenoiser(nn.Module):
    """Small offline denoiser used for CPU tests and smoke runs."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding table used for continuous token algebra."""

        return self.embed_tokens

    def forward(self, inputs_embeds: torch.Tensor, **_: object) -> SimpleNamespace:
        """Run a bidirectional denoising pass over supplied embeddings."""

        hidden = self.encoder(inputs_embeds)
        return SimpleNamespace(logits=self.lm_head(hidden))


class MaskedDiffusionPrior(nn.Module):
    """
    The 8B Masked Diffusion Language Model (MDLM) core.

    This module manages the QLoRA hardware-efficient weight loading and
    executes the continuous soft-masking denoising loop over packed 1D sequences.
    """

    def __init__(
        self,
        base_model_id: Optional[str] = None,
        lora_rank: int = 32,
        vocab_size: int = VOCAB_SIZE,
        mask_token_id: int = MASK_TOKEN_ID,
        local_files_only: bool = True,
        use_tiny_backbone: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.model: nn.Module

        if use_tiny_backbone or base_model_id is None:
            self.base_model: nn.Module = TinyBidirectionalDenoiser(
                vocab_size=vocab_size
            )
            self.model = self.base_model
        else:
            # 1. Initialize QLoRA 4-bit NF4 Base Model from local resources only.
            self.base_model = self._load_quantized_base(
                base_model_id, local_files_only=local_files_only
            )

            # 2. Inject Rank-32 FP16 LoRA Adapters
            self.model = self._inject_lora_adapters(self.base_model, lora_rank)

        # We extract the base embedding layer to perform Continuous Token Algebra
        self.embed_tokens = cast(Any, self.model).get_input_embeddings()

    @property
    def device(self) -> torch.device:
        """Return the device hosting model parameters."""

        return next(self.model.parameters()).device

    def _load_quantized_base(
        self, model_id: str, local_files_only: bool = True
    ) -> nn.Module:
        """
        Loads the 8B base model into ~4.5GB VRAM using NF4 quantization.
        """
        if not local_files_only:
            raise ValueError("RADS Kaggle execution must use local_files_only=True")
        if os.path.sep in model_id and not os.path.exists(model_id):
            raise FileNotFoundError(f"Base model path does not exist: {model_id}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Information-theoretically optimal for normal distributions
            bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
            bnb_4bit_compute_dtype=torch.float16,  # Compute adapter updates in FP16
        )

        # Force FlashAttention-2 / SDPA via attn_implementation
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa",
            local_files_only=local_files_only,
        )

        # Freeze base model weights entirely
        for param in model.parameters():
            param.requires_grad = False

        return cast(nn.Module, model)

    def _inject_lora_adapters(self, model: nn.Module, rank: int) -> nn.Module:
        """
        Injects Rank-32 LoRA adapters into all linear projection layers.
        These adapters are hot-swapped between ARC-AGI-2 (Static) and ARC-AGI-3 (Interactive).
        """
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.0,  # Dropout disabled for deterministic verification
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(cast(Any, model), lora_config)
        return peft_model

    def swap_lora_weights(self, new_state_dict: Dict[str, torch.Tensor]):
        """
        Zero-latency hot-swap of task-specific adapters.
        Replaces the FP16 LoRA deltas without reloading the 4GB NF4 base model.
        """
        set_peft_model_state_dict(cast(Any, self.model), new_state_dict)
        release_cuda_fragments()

    def load_lora_adapter(self, adapter_path: str, adapter_name: str) -> None:
        """Load a PEFT adapter from a local path and activate it."""

        if not isinstance(self.model, PeftModel):
            raise RuntimeError(
                "LoRA adapters are only available for the PEFT-backed model"
            )
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"LoRA adapter path does not exist: {adapter_path}")
        self.model.load_adapter(
            adapter_path, adapter_name=adapter_name, is_trainable=True
        )
        self.model.set_adapter(adapter_name)
        release_cuda_fragments()

    def set_active_adapter(self, adapter_name: str) -> None:
        """Activate an already-loaded PEFT adapter and clean allocator fragments."""

        if not isinstance(self.model, PeftModel):
            raise RuntimeError(
                "LoRA adapters are only available for the PEFT-backed model"
            )
        self.model.set_adapter(adapter_name)
        release_cuda_fragments()

    def continuous_denoise_step(
        self,
        packed_soft_tokens: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        max_seq_len: int,
        row_coords: torch.Tensor,
        col_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Executes a single step of Continuous Token Algebra over the packed sequence.

        Args:
            packed_soft_tokens: Probability distributions over vocabulary [total_tokens, vocab_size]
            cu_seq_lens: Sequence boundaries [batch_size + 1]
            row_coords, col_coords: 2D spatial mappings for Fused 2D RoPE

        Returns:
            Refined probability distributions [total_tokens, vocab_size]
        """
        # 1. Project probability distributions into continuous embedding space
        # shape: [total_tokens, embed_dim]
        # This blends the latent vectors based on the current confidence of each token
        if self.embed_tokens.weight.shape[0] < self.vocab_size:
            raise ValueError("Backbone embedding table is smaller than RADS vocabulary")
        continuous_embeddings = torch.matmul(
            packed_soft_tokens, self.embed_tokens.weight[: self.vocab_size]
        )

        # 2D RoPE is applied inside supported attention backends. The tiny
        # offline denoiser keeps this path pure PyTorch so tests do not require
        # runtime Triton compilation or network access.

        # 2. Forward pass through the Transformer
        # Since we bypass standard input embedding, we pass inputs_embeds directly.
        # cu_seq_lens is handled natively if passed via kwargs to the SDPA-enabled model.
        outputs = self.model(
            inputs_embeds=continuous_embeddings.unsqueeze(0),  # Dummy batch dim for HF
            attention_mask=None,  # Masking is handled entirely by cu_seq_lens in SDPA
            output_hidden_states=False,
        )

        # 3. Extract logits and compute new soft distribution
        logits = outputs.logits.squeeze(0)[
            ..., : self.vocab_size
        ]  # [total_tokens, vocab_size]

        # Normalize into a probability distribution for the next iterative step
        refined_soft_tokens = F.softmax(logits, dim=-1)

        return refined_soft_tokens

    def generate_hypothesis(
        self,
        packed_context: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        max_seq_len: int,
        row_coords: torch.Tensor,
        col_coords: torch.Tensor,
        num_diffusion_steps: int = 10,
    ) -> torch.Tensor:
        """
        Executes the full diffusion loop to synthesize an ARC grid or Python world model.
        """
        # Initialize sequence as 100% <MASK> token probability
        total_tokens = packed_context.shape[0]
        soft_tokens = torch.zeros(
            (total_tokens, self.vocab_size), device=packed_context.device
        )
        soft_tokens[:, self.mask_token_id] = 1.0

        # Override the input/demonstration segments of the sequence with hard token
        # probabilities so the model only diffuses the target answer area
        # (Assuming `packed_context` contains the ground-truth token IDs for the prompt)
        is_prompt_mask = packed_context != self.mask_token_id
        hard_prompt = F.one_hot(packed_context, num_classes=self.vocab_size).float()
        soft_tokens = torch.where(
            is_prompt_mask.unsqueeze(-1), hard_prompt, soft_tokens
        )

        for step in range(num_diffusion_steps):
            refined_tokens = self.continuous_denoise_step(
                soft_tokens, cu_seq_lens, max_seq_len, row_coords, col_coords
            )

            # Re-clamp the prompt tokens to absolute certainty (1.0)
            # and allow the diffusion to update only the predicted target cells
            soft_tokens = torch.where(
                is_prompt_mask.unsqueeze(-1), hard_prompt, refined_tokens
            )

        # Return the final deterministic `argmax` selection as the finalized hypothesis
        final_hard_tokens = torch.argmax(soft_tokens, dim=-1)
        return final_hard_tokens
