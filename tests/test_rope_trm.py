import torch

from models.rope_2d import Fused2DRoPE
from models.trm_verifier import TinyRecursiveVerifier


def test_2d_rope_identity_at_origin_and_shape():
    rope = Fused2DRoPE(head_dim=8, max_grid_size=4)
    q = torch.randn(1, 3, 2, 8)
    k = torch.randn(1, 3, 2, 8)
    coords = torch.zeros(3, dtype=torch.long)

    q_rot, k_rot = rope(q, k, coords, coords)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert torch.allclose(q_rot, q)
    assert torch.allclose(k_rot, k)


def test_2d_rope_requires_axis_pair_dimensions():
    try:
        Fused2DRoPE(head_dim=6)
    except ValueError as exc:
        assert "divisible by 4" in str(exc)
    else:
        raise AssertionError("Expected invalid head_dim to raise ValueError")


def test_trm_trace_converges_when_recursive_block_is_zero():
    trm = TinyRecursiveVerifier(embed_dim=8, hidden_dim=16)
    for module in trm.recursive_block.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.zeros_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    z = torch.randn(4, 8)
    trace = trm(z, max_steps=5, epsilon=0.01)

    assert trace.logits.shape == (4, 1)
    assert trace.final_z.shape == z.shape
    assert trace.prev_z.shape == z.shape
    assert trace.delta_history.shape == (5, 4)
    assert trace.converged.all()
    assert torch.allclose(trace.final_delta, torch.zeros(4))
