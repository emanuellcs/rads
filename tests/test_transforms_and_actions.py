import numpy as np

from agent.physics_simulator import (
    ARCGameState,
    ARCPhysicsSimulator,
    compile_dummy_hypothesis,
)
from core.constants import ACTION1, ACTION2, ACTION3, ACTION4, ACTION6, RESET_ACTION
from data.transforms import apply_paired_color_permutation


def test_paired_color_permutation_preserves_cross_grid_relationships():
    inp = np.array([[1, 0], [0, 2]], dtype=np.uint8)
    out = np.array([[2, 0], [0, 1]], dtype=np.uint8)

    aug_inp, aug_out = apply_paired_color_permutation(
        inp, out, preserve_background=True, num_colors=10
    )

    assert aug_inp[0, 1] == 0
    assert aug_out[0, 1] == 0
    assert aug_out[0, 0] == aug_inp[1, 1]
    assert aug_out[1, 1] == aug_inp[0, 0]


def test_dummy_physics_uses_normalized_action_ids():
    grid = np.zeros((3, 3), dtype=np.uint8)
    simulator = ARCPhysicsSimulator(compile_dummy_hypothesis)
    state = ARCGameState(grid=grid, agent_r=1, agent_c=1)

    assert RESET_ACTION == 0
    assert simulator.step(state, ACTION1).agent_c == 2
    assert simulator.step(state, ACTION2).agent_c == 0
    assert simulator.step(state, ACTION3).agent_r == 2
    assert simulator.step(state, ACTION4).agent_r == 0

    toggled = simulator.step(state, ACTION6)
    assert toggled.grid[1, 1] == 1
