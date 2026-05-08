from __future__ import annotations

import numpy as np

from core.constants import ARC_AGI_3_COLORS


def sample_color_permutation(
    grids: tuple[np.ndarray, ...],
    preserve_background: bool = True,
    num_colors: int = ARC_AGI_3_COLORS,
) -> np.ndarray:
    """
    Sample one bijective color permutation for a set of related grids.

    The same mapping must be used for input and output grids. Applying
    independent permutations would change the demonstrated transformation
    itself and poison both pre-training and test-time training targets.
    """

    if num_colors <= 0:
        raise ValueError("num_colors must be positive")

    palette = np.arange(num_colors, dtype=np.uint8)
    unique = np.unique(np.concatenate([grid.reshape(-1) for grid in grids]))
    if preserve_background:
        active = unique[unique != 0]
    else:
        active = unique

    if active.size > 1:
        shuffled = np.random.permutation(active)
        palette[active] = shuffled.astype(np.uint8)

    return palette


def apply_color_permutation(
    grid: np.ndarray,
    preserve_background: bool = True,
    permutation: np.ndarray | None = None,
    num_colors: int = ARC_AGI_3_COLORS,
) -> np.ndarray:
    """
    Applies a random bijective permutation to the color palette of the grid.
    This forces the Diffusion Prior to learn the topological rules (e.g., "fill the enclosed area")
    rather than memorizing specific color mappings (e.g., "turn red to blue").

    Args:
        grid: 2D NumPy array representing the ARC grid (0-9 for ARC-AGI-2, 0-15 for ARC-AGI-3).
        preserve_background: If True, color 0 (traditionally black/background) is not permuted.
                             This is often beneficial for spatial grounding in ARC tasks.

    Returns:
        A new 2D NumPy array with permuted colors.
    """
    if permutation is None:
        permutation = sample_color_permutation((grid,), preserve_background, num_colors)
    if int(grid.max(initial=0)) >= len(permutation):
        raise ValueError("Grid contains a color outside the supplied permutation")

    # Return a newly allocated array to prevent mutating shared memory references
    return permutation[grid].astype(grid.dtype, copy=True)


def apply_paired_color_permutation(
    inp_grid: np.ndarray,
    out_grid: np.ndarray,
    preserve_background: bool = True,
    num_colors: int = ARC_AGI_3_COLORS,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one sampled color permutation consistently to an input/output pair."""

    permutation = sample_color_permutation(
        (inp_grid, out_grid), preserve_background, num_colors
    )
    return (
        apply_color_permutation(inp_grid, preserve_background, permutation, num_colors),
        apply_color_permutation(out_grid, preserve_background, permutation, num_colors),
    )


def apply_rotation(grid: np.ndarray, k: int) -> np.ndarray:
    """
    Rotates the grid by 90 degrees * k.

    Args:
        grid: 2D NumPy array.
        k: Integer multiplier for 90-degree rotations (e.g., 1=90, 2=180, 3=270).

    Returns:
        A new 2D NumPy array.
    """
    # np.rot90 returns a view. We explicitly call .copy() to ensure the worker
    # process owns this memory completely, preventing any upstream CoW leaks.
    return np.rot90(grid, k=k).copy()


def apply_reflection(grid: np.ndarray, axis: str) -> np.ndarray:
    """
    Flips the grid along the specified axis.

    Args:
        grid: 2D NumPy array.
        axis: 'h' for horizontal (left-right) flip, 'v' for vertical (up-down) flip.

    Returns:
        A new 2D NumPy array.
    """
    if axis == "h":
        return np.fliplr(grid).copy()
    elif axis == "v":
        return np.flipud(grid).copy()
    else:
        raise ValueError(f"Invalid reflection axis: '{axis}'. Use 'h' or 'v'.")


def apply_random_symmetry_group(
    inp_grid: np.ndarray, out_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies an identical, randomly selected D4 symmetry group transformation
    (rotation + reflection) to both the input and output grids simultaneously.

    This ensures the geometric relationship between the input and output
    remains mathematically consistent for Test-Time Training (TTT).
    """
    # 1. Random Rotation (0, 90, 180, or 270 degrees)
    k = np.random.randint(0, 4)
    if k > 0:
        inp_grid = apply_rotation(inp_grid, k)
        out_grid = apply_rotation(out_grid, k)

    # 2. Random Reflection (None, Horizontal, or Vertical)
    reflection_choices: tuple[str | None, ...] = (None, "h", "v")
    reflection_choice = reflection_choices[
        int(np.random.randint(0, len(reflection_choices)))
    ]
    if reflection_choice is not None:
        inp_grid = apply_reflection(inp_grid, reflection_choice)
        out_grid = apply_reflection(out_grid, reflection_choice)

    return inp_grid, out_grid
