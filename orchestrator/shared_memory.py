from __future__ import annotations

from multiprocessing import shared_memory
import queue
import numpy as np
import torch.multiprocessing as mp
from typing import Any, Dict, List

# Enforce the 'spawn' start method for PyTorch multiprocessing.
# 'fork' causes CUDA context corruption and Copy-on-Write memory leaks.
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


class IPCMemoryManager:
    """
    Manages the shared memory segments and communication queues for the
    Asynchronous MCTS / GPU Batch Server architecture.
    """

    def __init__(
        self, num_slots: int = 256, state_bytes: int = 4096, num_workers: int = 8
    ):
        """
        Allocates the shared memory buffers and IPC queues.

        Args:
            num_slots: Maximum number of concurrent neural evaluations in flight.
            state_bytes: Maximum byte size of a serialized ARC-AGI-3 game state.
            num_workers: Number of CPU MCTS workers.
        """
        self.num_slots = num_slots
        self.state_bytes = state_bytes
        self.num_workers = num_workers
        if state_bytes % np.dtype(np.float32).itemsize != 0:
            raise ValueError(
                "state_bytes must be divisible by 4 for float32 shared slots"
            )
        self.latent_dim = state_bytes // np.dtype(np.float32).itemsize

        # 1. Allocate Shared Memory Buffers (Zero-Copy across processes)
        self.states_shm = shared_memory.SharedMemory(
            create=True,
            size=num_slots * self.latent_dim * np.dtype(np.float32).itemsize,
        )
        self.scores_shm = shared_memory.SharedMemory(
            create=True,
            size=num_slots * np.dtype(np.float32).itemsize,
        )
        self.states = np.ndarray(
            (num_slots, self.latent_dim), dtype=np.float32, buffer=self.states_shm.buf
        )
        self.scores = np.ndarray(
            (num_slots,), dtype=np.float32, buffer=self.scores_shm.buf
        )
        self.states.fill(0.0)
        self.scores.fill(0.0)

        # 2. Allocate IPC Queues
        # Workers push (worker_id, slot_id) here to request a GPU evaluation
        self.request_queue: Any = mp.Queue()

        # GPU pushes the completed slot_id back to the specific worker's queue
        self.result_queues: Dict[int, Any] = {i: mp.Queue() for i in range(num_workers)}

        # 3. Slot Management (Collision Prevention)
        # Thread-safe queue containing all available slot indices [0, 1, ..., num_slots - 1]
        self.available_slots: Any = mp.Queue()
        for i in range(num_slots):
            self.available_slots.put(i)

    def get_worker_interfaces(self) -> List[Dict]:
        """
        Returns a list of interface dictionaries, one for each CPU worker.
        These contain only the specific queues and memory pointers that worker needs.
        """
        interfaces = []
        for worker_id in range(self.num_workers):
            interfaces.append(
                {
                    "worker_id": worker_id,
                    "request_queue": self.request_queue,
                    "result_queue": self.result_queues[worker_id],
                    "available_slots": self.available_slots,
                    "states_shm_name": self.states_shm.name,
                    "scores_shm_name": self.scores_shm.name,
                    "state_bytes": self.state_bytes,
                    "num_slots": self.num_slots,
                    "latent_dim": self.latent_dim,
                }
            )
        return interfaces

    def get_gpu_server_interface(self) -> Dict:
        """
        Returns the interface dictionary for the GPU Batch Server process.
        """
        return {
            "request_queue": self.request_queue,
            "result_queues": self.result_queues,
            "states_shm_name": self.states_shm.name,
            "scores_shm_name": self.scores_shm.name,
            "state_bytes": self.state_bytes,
            "num_slots": self.num_slots,
            "latent_dim": self.latent_dim,
        }

    def close(self, unlink: bool = True) -> None:
        """Close and optionally unlink owned shared-memory segments."""

        for shm in (self.states_shm, self.scores_shm):
            try:
                shm.close()
            finally:
                if unlink:
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass


# ==========================================
# Worker-Side Helper Functions
# ==========================================


class IPCWorkerClient:
    """
    A lightweight client instantiated inside each CPU worker process to interact
    with the shared memory buffers safely.
    """

    def __init__(self, interface: Dict):
        self.worker_id = interface["worker_id"]
        self.request_queue = interface["request_queue"]
        self.result_queue = interface["result_queue"]
        self.available_slots = interface["available_slots"]
        self.state_bytes = interface["state_bytes"]
        self.num_slots = interface["num_slots"]
        self.latent_dim = interface["latent_dim"]

        self._states_shm = shared_memory.SharedMemory(name=interface["states_shm_name"])
        self._scores_shm = shared_memory.SharedMemory(name=interface["scores_shm_name"])
        self._states_buffer = np.ndarray(
            (self.num_slots, self.latent_dim),
            dtype=np.float32,
            buffer=self._states_shm.buf,
        )
        self._scores_buffer = np.ndarray(
            (self.num_slots,), dtype=np.float32, buffer=self._scores_shm.buf
        )

    def evaluate_state(
        self, serialized_state: np.ndarray, timeout_s: float | None = 30.0
    ) -> float:
        """
        Synchronous wrapper around the async IPC architecture.
        The CPU worker calls this, writes to memory, and sleeps until the GPU
        notifies it that the score is ready.
        """
        state = np.asarray(serialized_state, dtype=np.float32).reshape(-1)
        if state.size > self.latent_dim:
            raise ValueError(
                f"State length {state.size} exceeds latent dim {self.latent_dim}"
            )

        # 1. Checkout an available memory slot (blocks if GPU is completely backed up)
        slot_id = self.available_slots.get()

        try:
            # 2. Write a fully deterministic slot. Zeroing prevents stale tail values
            # from a larger previous state contaminating a smaller current state.
            self._states_buffer[slot_id].fill(0.0)
            self._states_buffer[slot_id, : state.size] = state

            # 3. Notify the GPU Batch Server
            self.request_queue.put((self.worker_id, slot_id))

            # 4. Yield the CPU and sleep until the GPU finishes the batch and pings us back
            try:
                returned_slot_id = self.result_queue.get(timeout=timeout_s)
            except queue.Empty as exc:
                raise TimeoutError(
                    f"Timed out waiting for GPU score in slot {slot_id}"
                ) from exc
            if returned_slot_id != slot_id:
                raise RuntimeError("IPC routing error: received mismatched slot ID.")

            # 5. Read the float score calculated by the TRM
            return float(self._scores_buffer[slot_id])
        finally:
            # 6. Return the slot to the available pool even when worker-side errors occur.
            self.available_slots.put(slot_id)

    def close(self) -> None:
        """Detach this worker from shared-memory segments."""

        self._states_shm.close()
        self._scores_shm.close()
