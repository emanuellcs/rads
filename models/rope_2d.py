from typing import Tuple, cast

import torch
import torch.nn as nn


class Fused2DRoPE(nn.Module):
    """
    Factorized 2D Rotary Positional Encoding (RoPE).

    Splits the head dimension into two axis partitions. Applies standard
    1D RoPE to the row partition and independently to the column partition.
    """

    def __init__(self, head_dim: int, max_grid_size: int = 64, base: float = 10000.0):
        """
        Args:
            head_dim: The dimension of each attention head. Must be divisible by 4.
            max_grid_size: The maximum dimension of the ARC grid (64 for AGI-3).
            base: The base for the exponential frequency decay.
        """
        super().__init__()

        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim must be strictly divisible by 4 for axial 2D RoPE, got {head_dim}"
            )

        self.head_dim = head_dim
        self.half_dim = head_dim // 2
        self.max_grid_size = max_grid_size
        self.base = base

        # Precompute the inverse frequencies for the half-dimension
        # theta_i = 10000^(-2(i-1)/d)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-cache the cosine and sine matrices for the maximum possible grid size
        # to prevent recomputing these trigonometric functions during the forward pass.
        self._precompute_freqs_cis()

    def _precompute_freqs_cis(self):
        """
        Caches the cos and sin frequencies for absolute grid positions up to max_grid_size.
        """
        t = torch.arange(
            self.max_grid_size, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )

        # freqs shape: [max_grid_size, half_dim / 2]
        freqs = torch.outer(t, self.inv_freq)

        # Duplicate frequencies to match the complex rotation pattern
        # emb shape: [max_grid_size, half_dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cached_cos", emb.cos(), persistent=False)
        self.register_buffer("cached_sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotates half the hidden dims of the input.
        [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        row_coords: torch.Tensor,
        col_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies 2D RoPE to queries and keys based on their spatial grid coordinates.

        Args:
            q: Query tensor. Shape: [1, total_tokens, num_heads, head_dim]
            k: Key tensor. Shape: [1, total_tokens, num_heads, head_dim]
            row_coords: 1D tensor of row indices [total_tokens]
            col_coords: 1D tensor of column indices [total_tokens]

        Returns:
            q_rot, k_rot: The rotated query and key tensors.
        """
        # q and k are packed sequences, so batch dim is 1
        # shape expected: [1, total_tokens, num_heads, head_dim]
        squeeze_batch = False
        if q.dim() == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            squeeze_batch = True
        if q.dim() != 4 or k.dim() != 4:
            raise ValueError("q and k must have shape [batch, tokens, heads, head_dim]")
        if q.shape != k.shape:
            raise ValueError("q and k must have identical shapes")

        # Ensure coordinates are safely bounded
        cached_cos = cast(torch.Tensor, self.cached_cos)
        cached_sin = cast(torch.Tensor, self.cached_sin)
        row_coords = row_coords.to(device=cached_cos.device, dtype=torch.long).clamp(
            0, self.max_grid_size - 1
        )
        col_coords = col_coords.to(device=cached_cos.device, dtype=torch.long).clamp(
            0, self.max_grid_size - 1
        )

        # Look up cached cos/sin for row and column positions
        # shapes: [total_tokens, half_dim]
        cos_row = cached_cos[row_coords]
        sin_row = cached_sin[row_coords]

        cos_col = cached_cos[col_coords]
        sin_col = cached_sin[col_coords]

        # Reshape to broadcast across heads
        # shapes: [1, total_tokens, 1, half_dim]
        cos_row = cos_row.unsqueeze(0).unsqueeze(2)
        sin_row = sin_row.unsqueeze(0).unsqueeze(2)
        cos_col = cos_col.unsqueeze(0).unsqueeze(2)
        sin_col = sin_col.unsqueeze(0).unsqueeze(2)

        q_row, q_col = q.split(self.half_dim, dim=-1)
        k_row, k_col = k.split(self.half_dim, dim=-1)

        q_row = (q_row * cos_row) + (self._rotate_half(q_row) * sin_row)
        q_col = (q_col * cos_col) + (self._rotate_half(q_col) * sin_col)
        k_row = (k_row * cos_row) + (self._rotate_half(k_row) * sin_row)
        k_col = (k_col * cos_col) + (self._rotate_half(k_col) * sin_col)

        q_embed = torch.cat([q_row, q_col], dim=-1)
        k_embed = torch.cat([k_row, k_col], dim=-1)

        if squeeze_batch:
            return q_embed.squeeze(0), k_embed.squeeze(0)
        return q_embed, k_embed


# ==========================================
# Integration Hook for diffusion_prior.py
# ==========================================


def inject_2d_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    row_coords: torch.Tensor,
    col_coords: torch.Tensor,
    rope_module: Fused2DRoPE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hook to apply 2D RoPE inside the continuous denoising step.
    In the final Kaggle environment, this wrapper will route directly to
    the Unsloth fused Triton kernel to save VRAM reads/writes.
    """
    return rope_module(q, k, row_coords, col_coords)
