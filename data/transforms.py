import numpy as np

def apply_color_permutation(grid: np.ndarray, preserve_background: bool = True) -> np.ndarray:
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
    # Identify unique colors present in the grid
    unique_colors = np.unique(grid)
    
    if preserve_background and 0 in unique_colors:
        # Exclude 0 from the permutation pool
        active_colors = unique_colors[unique_colors != 0]
        permuted_colors = np.random.permutation(active_colors)
        
        # Build the mapping dictionary
        color_map = {0: 0}
        color_map.update({old: new for old, new in zip(active_colors, permuted_colors)})
    else:
        permuted_colors = np.random.permutation(unique_colors)
        color_map = {old: new for old, new in zip(unique_colors, permuted_colors)}
        
    # Vectorized application of the color map using NumPy advanced indexing.
    # We create a palette array that covers the maximum possible color integer.
    max_color = grid.max()
    palette = np.arange(max_color + 1, dtype=grid.dtype)
    
    for old_color, new_color in color_map.items():
        palette[old_color] = new_color
        
    # Return a newly allocated array to prevent mutating shared memory references
    return palette[grid].copy()


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
    if axis == 'h':
        return np.fliplr(grid).copy()
    elif axis == 'v':
        return np.flipud(grid).copy()
    else:
        raise ValueError(f"Invalid reflection axis: '{axis}'. Use 'h' or 'v'.")


def apply_random_symmetry_group(inp_grid: np.ndarray, out_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    reflection_choice = np.random.choice([None, 'h', 'v'])
    if reflection_choice is not None:
        inp_grid = apply_reflection(inp_grid, reflection_choice)
        out_grid = apply_reflection(out_grid, reflection_choice)
        
    return inp_grid, out_grid