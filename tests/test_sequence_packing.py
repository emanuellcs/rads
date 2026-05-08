import torch

from models.sequence_packing import GridSequencePacker, execute_packed_attention


def test_pack_unpack_mixed_grid_shapes():
    grids = [
        torch.tensor([[1]]),
        torch.arange(6).view(2, 3),
        torch.arange(5),
    ]

    packed = GridSequencePacker.pack_grids(grids)

    assert packed["packed_sequence"].tolist() == [1, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4]
    assert packed["cu_seq_lens"].tolist() == [0, 1, 7, 12]
    assert packed["max_seq_len"] == 6
    assert packed["row_coords"].shape == packed["col_coords"].shape == (12,)

    unpacked = GridSequencePacker.unpack_sequence(
        packed["packed_sequence"], packed["grid_shapes"]
    )
    assert torch.equal(unpacked[0], grids[0])
    assert torch.equal(unpacked[1], grids[1])
    assert torch.equal(unpacked[2], grids[2])


def test_execute_packed_attention_matches_per_sequence_sdpa_cpu():
    torch.manual_seed(0)
    lengths = [2, 3, 1]
    cu = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    q = torch.randn(sum(lengths), 2, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    packed_out = execute_packed_attention(q, k, v, cu, max_seq_len=3)

    expected = []
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
        q_i = q[start:end].permute(1, 0, 2).unsqueeze(0)
        k_i = k[start:end].permute(1, 0, 2).unsqueeze(0)
        v_i = v[start:end].permute(1, 0, 2).unsqueeze(0)
        out_i = torch.nn.functional.scaled_dot_product_attention(
            q_i, k_i, v_i, dropout_p=0.0
        )
        expected.append(out_i.squeeze(0).permute(1, 0, 2))

    assert torch.allclose(packed_out, torch.cat(expected, dim=0), atol=1e-6)
