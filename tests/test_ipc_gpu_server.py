import threading
from types import SimpleNamespace

import numpy as np
import torch

from orchestrator.gpu_batch_server import GPUBatchServer, POISON_PILL
from orchestrator.shared_memory import IPCMemoryManager, IPCWorkerClient


class ShapeAssertingTRM(torch.nn.Module):
    def __init__(self, batch_size: int, latent_dim: int):
        super().__init__()
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.seen_shapes = []

    def forward(self, x: torch.Tensor):
        self.seen_shapes.append(tuple(x.shape))
        assert tuple(x.shape) == (self.batch_size, self.latent_dim)
        return SimpleNamespace(logits=x.sum(dim=1, keepdim=True))


def test_gpu_server_pads_partial_batch_to_static_shape():
    manager = IPCMemoryManager(num_slots=8, state_bytes=5 * 4, num_workers=2)
    fake_trm = ShapeAssertingTRM(batch_size=64, latent_dim=5)
    server = GPUBatchServer(
        manager.get_gpu_server_interface(),
        fake_trm,
        device="cpu",
        batch_size=64,
        flush_timeout_ms=5.0,
    )

    try:
        manager.states[0, :3] = [1.0, 2.0, 3.0]
        manager.states[1, :2] = [4.0, 5.0]
        manager.request_queue.put((0, 0))
        manager.request_queue.put((1, 1))

        thread = threading.Thread(target=server.serve_forever)
        thread.start()

        assert manager.result_queues[0].get(timeout=2.0) == 0
        assert manager.result_queues[1].get(timeout=2.0) == 1
        assert manager.scores[0] == 6.0
        assert manager.scores[1] == 9.0
        assert fake_trm.seen_shapes == [(64, 5)]

        manager.request_queue.put(POISON_PILL)
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    finally:
        manager.close()


def test_ipc_worker_client_zeroes_slot_tail_and_reuses_slot():
    manager = IPCMemoryManager(num_slots=1, state_bytes=4 * 4, num_workers=1)
    client = IPCWorkerClient(manager.get_worker_interfaces()[0])

    def responder():
        worker_id, slot_id = manager.request_queue.get(timeout=2.0)
        assert worker_id == 0
        assert slot_id == 0
        assert np.allclose(manager.states[0], [1.0, 2.0, 0.0, 0.0])
        manager.scores[slot_id] = 7.5
        manager.result_queues[worker_id].put(slot_id)

    thread = threading.Thread(target=responder)
    thread.start()

    try:
        score = client.evaluate_state(
            np.array([1.0, 2.0], dtype=np.float32), timeout_s=2.0
        )
        assert score == 7.5
        assert manager.available_slots.get(timeout=2.0) == 0
    finally:
        thread.join(timeout=2.0)
        client.close()
        manager.close()
