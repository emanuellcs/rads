import numpy as np


def generate() -> tuple[np.ndarray, np.ndarray]:
    """
    Procedural Generator: Connect the Dots (Horizontal/Vertical)

    Generates a grid with two dots of the same color. The output grid
    connects them with a line of the same color.

    API Contract:
    - MUST be strictly stateless (no global variables mutated).
    - MUST rely on the implicitly seeded `np.random` state configured by the
      DataLoader's `worker_init_fn` to guarantee orthogonal procedural generation.
    """
    # 1. Randomize grid dimensions (e.g., between 5x5 and 15x15)
    h = np.random.randint(5, 16)
    w = np.random.randint(5, 16)

    # Initialize blank input and output grids (color 0 = background)
    inp_grid = np.zeros((h, w), dtype=np.uint8)

    # 2. Pick a random drawing color (1-9)
    color = np.random.randint(1, 10)

    # 3. Determine alignment (0 = horizontal, 1 = vertical)
    is_horizontal = np.random.rand() > 0.5

    if is_horizontal:
        row = np.random.randint(0, h)
        col1, col2 = np.random.choice(w, 2, replace=False)
        start_c, end_c = min(col1, col2), max(col1, col2)

        # Place dots on the input grid
        inp_grid[row, start_c] = color
        inp_grid[row, end_c] = color

        # Draw the connecting line on the output grid
        out_grid = inp_grid.copy()
        out_grid[row, start_c : end_c + 1] = color

    else:
        col = np.random.randint(0, w)
        row1, row2 = np.random.choice(h, 2, replace=False)
        start_r, end_r = min(row1, row2), max(row1, row2)

        # Place dots on the input grid
        inp_grid[start_r, col] = color
        inp_grid[end_r, col] = color

        # Draw the connecting line on the output grid
        out_grid = inp_grid.copy()
        out_grid[start_r : end_r + 1, col] = color

    # The dataset pipeline will automatically apply random rotations,
    # reflections, and color permutations to this base structure.
    return inp_grid, out_grid
