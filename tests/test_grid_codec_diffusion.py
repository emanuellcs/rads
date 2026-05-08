import torch

from core.constants import MASK_TOKEN_ID, SEP_TOKEN_ID, VOCAB_SIZE
from core.grid_codec import (
    build_io_prompt,
    build_masked_output_prompt,
    dummy_prediction_from_input,
    encode_grid,
    one_hot_tokens,
)
from models.diffusion_prior import MaskedDiffusionPrior
from models.sequence_packing import GridSequencePacker


def test_grid_codec_builds_prompts_and_one_hot_tokens():
    grid = [[1, 2], [3, 4]]
    target = [[4, 3], [2, 1]]

    encoded = encode_grid(grid)
    prompt, slices = build_io_prompt(grid, target)
    masked, masked_slices = build_masked_output_prompt(grid, (2, 2))
    one_hot = one_hot_tokens(prompt)

    assert encoded.shape == (2, 2)
    assert prompt.tolist() == [1, 2, 3, 4, SEP_TOKEN_ID, 4, 3, 2, 1]
    assert slices.input_start == 0
    assert slices.input_end == 4
    assert slices.target_start == 5
    assert slices.target_end == 9
    assert masked.tolist() == [
        1,
        2,
        3,
        4,
        SEP_TOKEN_ID,
        MASK_TOKEN_ID,
        MASK_TOKEN_ID,
        MASK_TOKEN_ID,
        MASK_TOKEN_ID,
    ]
    assert masked_slices.target_start == 5
    assert one_hot.shape == (9, VOCAB_SIZE)
    assert dummy_prediction_from_input(grid) == grid


def test_grid_codec_rejects_invalid_shapes_and_colors():
    try:
        encode_grid(torch.tensor([1, 2, 3]))
    except ValueError as exc:
        assert "2D grid" in str(exc)
    else:
        raise AssertionError("Expected invalid grid rank to fail")

    try:
        encode_grid([[99]])
    except ValueError as exc:
        assert "Grid color IDs" in str(exc)
    else:
        raise AssertionError("Expected invalid color ID to fail")


def test_tiny_diffusion_path_runs_offline_without_model_weights():
    model = MaskedDiffusionPrior(use_tiny_backbone=True)
    prompt, _ = build_masked_output_prompt([[1, 2]], (1, 2))
    packed = GridSequencePacker.pack_grids([prompt])

    hard = model.generate_hypothesis(
        packed["packed_sequence"],
        packed["cu_seq_lens"],
        packed["max_seq_len"],
        packed["row_coords"],
        packed["col_coords"],
        num_diffusion_steps=1,
    )

    assert hard.shape == prompt.shape
    assert hard[:3].tolist() == [1, 2, SEP_TOKEN_ID]
    assert hard.max().item() < VOCAB_SIZE
