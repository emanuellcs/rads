import time
import queue
import torch
import numpy as np
from typing import Dict, Any
import multiprocessing as mp
from multiprocessing import shared_memory

# Define a distinct poison pill to safely shut down the infinite loop
POISON_PILL = (-1, -1)


class GPUBatchServer:
    """
    Dedicated GPU inference process for Asynchronous MCTS.

    This server continuously drains the Inter-Process Communication (IPC) request queue,
    dynamically batches serialized game states, executes the compiled Tiny Recursive Model (TRM),
    and returns the thermodynamic stability scores to the sleeping CPU workers.
    """

    def __init__(
        self,
        interface: Dict[str, Any],
        compiled_trm: torch.nn.Module,
        device: str = "cuda",
        batch_size: int = 64,
        flush_timeout_ms: float = 10.0,
    ):
        """
        Args:
            interface: The IPC dictionary from IPCMemoryManager.get_gpu_server_interface().
            compiled_trm: The torch.compile() wrapped TRM verifier.
            device: The target hardware device (e.g., "cuda:0").
            batch_size: The maximum dynamic batch size before forcing a forward pass.
            flush_timeout_ms: Maximum wait time (in milliseconds) before flushing a partial batch.
        """
        self.request_queue = interface["request_queue"]
        self.result_queues = interface["result_queues"]
        self.state_bytes = interface["state_bytes"]
        self.num_slots = interface["num_slots"]
        self.latent_dim = interface["latent_dim"]
        self.device = device

        self.compiled_trm = compiled_trm
        self.batch_size = batch_size
        self.flush_timeout_sec = flush_timeout_ms / 1000.0

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

    def _extract_state_tensor(self, slot_id: int) -> torch.Tensor:
        """
        Reads the serialized bytes for a specific memory slot and casts them to a Float32 tensor.
        """
        # Convert to PyTorch tensor without copying the underlying data (zero-copy)
        return torch.from_numpy(self._states_buffer[slot_id])

    def serve_forever(self):
        """
        The infinite polling loop. This function should be the entry point for
        the dedicated GPU multiprocessing.Process.
        """
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)

        # Pre-allocate a pinned memory buffer on the CPU to stage transfers to the GPU.
        # This allows for asynchronous, non-blocking Host-To-Device (H2D) PCIe transfers.
        # Assuming the state is a 1D vector; adjust dimensions based on actual TRM input.
        if device.type == "cuda":
            pinned_batch = torch.empty(
                (self.batch_size, self.latent_dim), dtype=torch.float32
            ).pin_memory()
        else:
            pinned_batch = torch.empty(
                (self.batch_size, self.latent_dim), dtype=torch.float32
            )
        static_device_batch = torch.empty(
            (self.batch_size, self.latent_dim), dtype=torch.float32, device=device
        )

        pending_requests = []

        print(
            f"[GPU Batch Server] Started on {self.device}. Awaiting MCTS evaluations..."
        )

        # Strictly enforce inference mode to disable autograd tracking completely,
        # saving VRAM and CPU cycles.
        with torch.inference_mode():
            while True:
                deadline = time.perf_counter() + self.flush_timeout_sec

                # 1. Dynamic Batch Accumulation
                while len(pending_requests) < self.batch_size:
                    timeout = max(0.0, deadline - time.perf_counter())
                    try:
                        # Blocks until a request arrives or the micro-timeout expires
                        req = self.request_queue.get(timeout=timeout)

                        if req == POISON_PILL:
                            print(
                                "[GPU Batch Server] Poison pill received. Shutting down."
                            )
                            self.close()
                            return

                        pending_requests.append(req)
                    except queue.Empty:
                        # Timeout reached. Break the accumulation loop to flush whatever we have.
                        break

                if not pending_requests:
                    continue

                # 2. Batch Construction (Zero-Copy Read -> Pinned Memory -> VRAM)
                current_batch_size = len(pending_requests)
                pinned_batch.zero_()
                for i, (worker_id, slot_id) in enumerate(pending_requests):
                    # Write the zero-copy view directly into the pinned memory buffer
                    pinned_batch[i].copy_(self._extract_state_tensor(slot_id))

                # Transfer the full static block to the GPU asynchronously.
                # The compiled TRM always sees [batch_size, latent_dim], even
                # when only the prefix contains real requests.
                static_device_batch.copy_(
                    pinned_batch, non_blocking=(device.type == "cuda")
                )

                # 3. Neural Execution (CUDA Graph Replay)
                trace = self.compiled_trm(static_device_batch)
                logits = trace.logits if hasattr(trace, "logits") else trace[0]
                scores = logits[:current_batch_size].squeeze(-1)

                # 4. Result Dispatch
                # Move the scores back to the CPU
                cpu_scores = scores.cpu().numpy()

                for i, (worker_id, slot_id) in enumerate(pending_requests):
                    # Write the float score to the shared RAM
                    self._scores_buffer[slot_id] = cpu_scores[i]

                    # Ping the specific CPU worker to wake it up
                    self.result_queues[worker_id].put(slot_id)

                # Clear the queue for the next batch cycle
                pending_requests.clear()

        self.close()

    def close(self) -> None:
        """Detach the server process from shared-memory segments."""

        self._states_shm.close()
        self._scores_shm.close()


# ==========================================
# Orchestrator Launch Utility
# ==========================================


def start_gpu_server_process(
    interface: Dict[str, Any], compiled_trm: torch.nn.Module
) -> mp.Process:
    """
    Spawns the GPU server in a dedicated background process.
    """
    server = GPUBatchServer(interface, compiled_trm)

    # Must use daemon=True so it automatically dies if the main orchestrator crashes
    p = mp.Process(target=server.serve_forever, daemon=True)
    p.start()
    return p
