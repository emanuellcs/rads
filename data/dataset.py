import random
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Dict

import numpy as np
import torch

from data.transforms import (
    apply_paired_color_permutation,
    apply_random_symmetry_group,
)

# ==========================================
# The CoW-Free Procedural Dataset
# ==========================================


class ARCDataset(Dataset):
    """
    A strictly stateless PyTorch Dataset for generating infinite ARC tasks.

    By keeping the generator_registry completely free of mutable Python objects
    (lists, large dicts, instances), we prevent the OS from triggering
    Copy-on-Write (CoW) page duplications during fork-based multiprocessing.
    """

    def __init__(
        self, generator_registry: Dict[str, Callable], virtual_size: int = 50_000_000
    ):
        """
        Args:
            generator_registry: A dictionary mapping concept names to pure Python functions
                                (e.g., RE-ARC procedural generators).
            virtual_size: The "epoch" length. Set massively high to stream continuously.
        """
        super().__init__()
        self.registry = generator_registry
        self.concept_names = list(generator_registry.keys())
        self.virtual_size = virtual_size

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Synthesizes a fresh task. All computation happens locally within the
        spawned worker process, meaning augmented tensors are allocated
        fresh in local memory and never dirty the parent process.
        """
        # 1. Uniformly sample a conceptual rule
        concept = self.concept_names[idx % len(self.concept_names)]

        # 2. Generate a base input/output pair using the stateless pure function
        # This function must return numpy arrays representing the grid integers
        input_grid, output_grid = self.registry[concept]()

        # 3. Apply CPU-Bound Stochastic Augmentations
        # Applied dynamically to maximize intra-batch variance
        if random.random() < 0.5:
            input_grid, output_grid = apply_paired_color_permutation(
                input_grid, output_grid
            )

        if random.random() < 0.5:
            input_grid, output_grid = apply_random_symmetry_group(
                input_grid, output_grid
            )

        # 4. Convert to PyTorch tensors
        # Note: We do NOT pad here. Padding is handled via Sequence Packing later.
        return {
            "input_grid": torch.tensor(input_grid.copy(), dtype=torch.long),
            "output_grid": torch.tensor(output_grid.copy(), dtype=torch.long),
            "concept_id": concept,
        }


# ==========================================
# Worker Initialization (RNG Isolation)
# ==========================================


def worker_init_fn(worker_id: int):
    """
    Called immediately after a worker process is forked.

    Derives a mathematically orthogonal seed for each worker to guarantee
    (1) No lock contention on shared RNGs.
    (2) Complete statistical independence of generated augmentations.
    """
    # Retrieve the base seed set by PyTorch in the main process
    base_seed = torch.initial_seed() % (2**31)

    # 31337 is an arbitrary prime offset to ensure distinct seed trajectories
    worker_seed = base_seed + worker_id * 31337

    # Isolate standard Python random state
    random.seed(worker_seed)

    # Isolate Numpy random state (used heavily by RE-ARC and geometric transforms)
    np.random.seed(worker_seed)


def create_arc_dataloader(
    generator_registry: Dict[str, Callable], batch_size: int = 64, num_workers: int = 4
) -> DataLoader:
    """Factory function to build the optimized DataLoader."""
    dataset = ARCDataset(generator_registry)
    persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,  # Enables async PCIe DMA transfers to GPU
        persistent_workers=persistent_workers,  # Avoids fork overhead when workers exist
        collate_fn=lambda x: (
            x
        ),  # Passthrough collator; sequence packing handles the batching
    )
