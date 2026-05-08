"""Tokenization helpers for ARC grids and prompt sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from core.constants import MASK_TOKEN_ID, SEP_TOKEN_ID, VOCAB_SIZE


@dataclass(frozen=True)
class PromptSlices:
    """Token index ranges for a packed input/target prompt."""

    input_start: int
    input_end: int
    target_start: int
    target_end: int


def encode_grid(
    grid: Sequence[Sequence[int]] | torch.Tensor, device: torch.device | None = None
) -> torch.Tensor:
    """Convert a rectangular ARC grid into a `torch.long` tensor."""

    tensor = (
        grid if isinstance(grid, torch.Tensor) else torch.tensor(grid, dtype=torch.long)
    )
    tensor = (
        tensor.to(device=device, dtype=torch.long)
        if device is not None
        else tensor.to(dtype=torch.long)
    )
    if tensor.dim() != 2:
        raise ValueError(f"Expected a 2D grid, got shape {tuple(tensor.shape)}")
    if tensor.numel() == 0:
        raise ValueError("ARC grids must contain at least one cell")
    if torch.any((tensor < 0) | (tensor >= MASK_TOKEN_ID)):
        raise ValueError("Grid color IDs must be in the range [0, 15]")
    return tensor


def build_io_prompt(
    input_grid: Sequence[Sequence[int]] | torch.Tensor,
    output_grid: Sequence[Sequence[int]] | torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, PromptSlices]:
    """Build `[input, SEP, output]` token IDs and return target slice metadata."""

    inp = encode_grid(input_grid, device=device).flatten()
    out = encode_grid(output_grid, device=device).flatten()
    sep = torch.tensor([SEP_TOKEN_ID], dtype=torch.long, device=inp.device)
    prompt = torch.cat([inp, sep, out], dim=0)
    slices = PromptSlices(
        0, inp.numel(), inp.numel() + 1, inp.numel() + 1 + out.numel()
    )
    return prompt, slices


def build_masked_output_prompt(
    input_grid: Sequence[Sequence[int]] | torch.Tensor,
    output_shape: tuple[int, int],
    device: torch.device | None = None,
) -> tuple[torch.Tensor, PromptSlices]:
    """Build `[input, SEP, MASK * output_cells]` for diffusion inference."""

    inp = encode_grid(input_grid, device=device).flatten()
    h, w = output_shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid output shape: {(h, w)}")
    sep = torch.tensor([SEP_TOKEN_ID], dtype=torch.long, device=inp.device)
    masked = torch.full((h * w,), MASK_TOKEN_ID, dtype=torch.long, device=inp.device)
    prompt = torch.cat([inp, sep, masked], dim=0)
    slices = PromptSlices(
        0, inp.numel(), inp.numel() + 1, inp.numel() + 1 + masked.numel()
    )
    return prompt, slices


def one_hot_tokens(tokens: torch.Tensor, vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """Return a float one-hot matrix for token IDs."""

    return torch.nn.functional.one_hot(
        tokens.to(torch.long), num_classes=vocab_size
    ).float()


def dummy_prediction_from_input(input_grid: Sequence[Sequence[int]]) -> list[list[int]]:
    """Return a valid fallback ARC prediction by repeating the test input grid."""

    return [list(map(int, row)) for row in input_grid]
