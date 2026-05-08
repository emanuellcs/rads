import numpy as np
from dataclasses import dataclass
from typing import Callable, List

from core.constants import (
    ACTION1,
    ACTION2,
    ACTION3,
    ACTION4,
    ACTION6,
    ARC_AGI_3_COLORS,
    MAX_GRID_SIZE,
    SIMPLE_ACTIONS,
)

# ARC-AGI-3 constraints
VOCAB_SIZE = ARC_AGI_3_COLORS  # Colors 0-15
SERIALIZED_STATE_FLOATS = MAX_GRID_SIZE * MAX_GRID_SIZE + 4
SERIALIZED_STATE_BYTES = SERIALIZED_STATE_FLOATS * np.dtype(np.float32).itemsize


@dataclass
class ARCGameState:
    """
    Lightweight, mutable data structure representing a single frame
    in the ARC-AGI-3 environment.
    """

    grid: np.ndarray  # 2D NumPy array of dtype=np.uint8 (0-15 colors)
    agent_r: int  # Agent's row coordinate (Y)
    agent_c: int  # Agent's column coordinate (X)
    is_terminal: bool = False
    is_win: bool = False

    def clone(self) -> "ARCGameState":
        """
        Creates a deep copy of the state for MCTS tree expansion.
        Using NumPy's copy() ensures independent memory allocation.
        """
        return ARCGameState(
            grid=self.grid.copy(),
            agent_r=self.agent_r,
            agent_c=self.agent_c,
            is_terminal=self.is_terminal,
            is_win=self.is_win,
        )


class ARCPhysicsSimulator:
    """
    The internal world model engine for the ARC-AGI-3 agent.

    This class takes a dynamically generated Python function (the hypothesis
    synthesized by the Diffusion Prior) and provides the MCTS-compliant API
    (step, is_terminal, serialize) needed to simulate future trajectories.
    """

    def __init__(
        self,
        rule_hypothesis_fn: Callable[[ARCGameState, int], ARCGameState],
        max_serialization_bytes: int = SERIALIZED_STATE_BYTES,
    ):
        """
        Args:
            rule_hypothesis_fn: A compiled Python function representing the
                                hypothesized transition dynamics T(S, A) -> S'.
            max_serialization_bytes: Must match the IPCMemoryManager configuration.
                                     Default stores a flattened 64x64 grid plus
                                     four float32 metadata values.
        """
        self.transition_model = rule_hypothesis_fn
        self.state_bytes = max_serialization_bytes
        self.latent_dim = self.state_bytes // 4  # 4 bytes per float32

    def get_valid_actions(self, state: ARCGameState) -> List[int]:
        """
        Returns the list of legal discrete actions.
        For internal MCTS planning, we exclude ACTION_RESET (which is handled
        by the Epistemic Forager as a meta-game exploit) and focus purely on
        the 7 standard movement/interaction actions.
        """
        if state.is_terminal:
            return []
        return list(SIMPLE_ACTIONS)

    def step(self, state: ARCGameState, action: int) -> ARCGameState:
        """
        Advances the internal world model by one tick using the injected
        hypothesis ruleset.
        """
        # 1. Clone the parent state to preserve the MCTS node integrity
        next_state = state.clone()

        # 2. Apply the dynamically synthesized Python rules
        # The transition_model modifies the next_state in-place or returns a new one.
        next_state = self.transition_model(next_state, action)

        # 3. Enforce hard environmental boundaries (Max 64x64)
        h, w = next_state.grid.shape
        next_state.agent_r = max(0, min(next_state.agent_r, h - 1))
        next_state.agent_c = max(0, min(next_state.agent_c, w - 1))

        return next_state

    def is_terminal(self, state: ARCGameState) -> bool:
        """Checks if the game has ended (Win or Hazard/Loss)."""
        return state.is_terminal

    def get_terminal_value(self, state: ARCGameState) -> float:
        """
        Scores the leaf node for MCTS backpropagation.
        ARC-AGI-3 is a single-agent puzzle, so returns are absolute.
        """
        if state.is_win:
            return 1.0
        elif state.is_terminal:
            # Game Over due to hazard / out-of-bounds
            return -1.0
        return 0.0

    def serialize_state(self, state: ARCGameState) -> np.ndarray:
        """
        Serializes the 2D game state into a fixed-size 1D float32 array for the
        Inter-Process Communication (IPC) buffer.

        This aligns perfectly with the GPU Batch Server's zero-copy requirements.
        """
        # 1. Flatten the variable-sized grid
        flat_grid = state.grid.astype(np.float32).flatten()

        # 2. Append critical metadata (Agent Pos, Grid Height, Grid Width)
        # We encode these as pseudo-tokens at the end of the tensor.
        h, w = state.grid.shape
        metadata = np.array([state.agent_r, state.agent_c, h, w], dtype=np.float32)

        combined = np.concatenate([flat_grid, metadata])

        # 3. Pad to the strict latent dimension required by the static CUDA graph
        current_len = combined.shape[0]
        if current_len > self.latent_dim:
            raise ValueError(
                f"Serialized state exceeds maximum dimension of {self.latent_dim}"
            )

        padded_state = np.zeros(self.latent_dim, dtype=np.float32)
        padded_state[:current_len] = combined

        return padded_state


# ==========================================
# Fallback / Baseline Hypothesis Generator
# ==========================================


def compile_dummy_hypothesis(state: ARCGameState, action: int) -> ARCGameState:
    """
    A baseline transition model for testing the MCTS and IPC pipelines before
    the Diffusion Prior has successfully inferred the true rules.

    Assumes standard 2D cardinal movement (Actions 1-4).
    """
    if action == ACTION1:  # RIGHT
        state.agent_c += 1
    elif action == ACTION2:  # LEFT
        state.agent_c -= 1
    elif action == ACTION3:  # DOWN
        state.agent_r += 1
    elif action == ACTION4:  # UP
        state.agent_r -= 1
    # Actions 5, 6, 7 are assumed to be non-spatial interactions (e.g., color toggle)
    elif action == ACTION6:
        # Dummy toggle color at current position
        current_color = state.grid[state.agent_r, state.agent_c]
        state.grid[state.agent_r, state.agent_c] = (current_color + 1) % VOCAB_SIZE

    return state
