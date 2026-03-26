import math
import numpy as np
from typing import List, Dict, Optional, Any

# Assuming standard ARC-AGI-3 action space: ACTION1-ACTION7 + RESET
# (0-6 are actions, 7 is RESET)
NUM_ACTIONS = 8

class MCTSNode:
    """
    A lightweight node in the Monte Carlo Search Tree.
    Optimized for fast updates and minimal memory overhead during deep searches.
    """
    __slots__ = [
        'state', 'parent', 'action_taken', 'children', 
        'visits', 'value_sum', 'is_terminal', 'is_expanded'
    ]
    
    def __init__(self, state: Any, parent: Optional['MCTSNode'] = None, action_taken: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        
        self.children: Dict[int, MCTSNode] = {}
        
        # MCTS Statistics
        self.visits: int = 0
        self.value_sum: float = 0.0
        
        # State flags determined by the Physics Simulator
        self.is_expanded: bool = False
        self.is_terminal: bool = False

    @property
    def q_value(self) -> float:
        """Average expected utility of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class MCTSAgent:
    """
    The asynchronous Monte Carlo Tree Search engine.
    
    Utilizes an internal Python physics replica for instantaneous tree expansion 
    and offloads thermodynamic stability evaluation to the GPU via IPC.
    """
    
    def __init__(self, 
                 ipc_client, 
                 physics_simulator, 
                 c_puct: float = 1.25):
        """
        Args:
            ipc_client: The IPCWorkerClient instance connected to the shared memory queues.
            physics_simulator: The internal Python replica of the ARC-AGI-3 environment.
            c_puct: The exploration constant (higher = more exploration).
        """
        self.ipc = ipc_client
        self.simulator = physics_simulator
        self.c_puct = c_puct

    def search(self, root_state: Any, num_simulations: int = 5000) -> int:
        """
        Executes the MCTS algorithm from the given root state.
        
        Args:
            root_state: The current state of the ARC-AGI-3 game.
            num_simulations: Number of tree expansion rollouts to perform.
            
        Returns:
            The optimal action index to take in the real environment.
        """
        root = MCTSNode(state=root_state)
        
        for _ in range(num_simulations):
            node = root
            
            # 1. Selection
            # Traverse down the tree picking the best UCB/PUCT action until a leaf is found.
            while node.is_expanded and not node.is_terminal:
                node = self._select_best_child(node)
                
            # 2. Expansion & Simulation
            # If the game isn't over, expand the node and ask the TRM to evaluate its stability.
            if not node.is_terminal:
                self._expand(node)
                
                # Check if the expansion immediately triggered a terminal state (Win/Loss)
                if node.is_terminal:
                    # Simulator assigns +1.0 for Win, -1.0 for Game Over / Hazard
                    value = self.simulator.get_terminal_value(node.state)
                else:
                    # Serialize the state to a contiguous byte array for the shared memory buffer
                    serialized_state = self.simulator.serialize_state(node.state)
                    
                    # IPC Call: Yields the CPU thread until the GPU returns the TRM score
                    value = self.ipc.evaluate_state(serialized_state)
            else:
                value = self.simulator.get_terminal_value(node.state)
                
            # 3. Backpropagation
            # Propagate the TRM's evaluation score back up the tree.
            self._backpropagate(node, value)
            
        # Extract the most robust action from the root based on visit count (not just raw Q-value)
        # to ensure resilience against statistical outliers.
        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_action

    def _select_best_child(self, node: MCTSNode) -> MCTSNode:
        """
        Selects the optimal child using the PUCT formula:
        PUCT(s, a) = Q(s, a) + c_puct * prior * sqrt(N(s)) / (1 + N(s, a))
        """
        best_score = -float('inf')
        best_child = None
        
        # Base exploration factor scaling with the parent's visit count
        sqrt_parent_visits = math.sqrt(node.visits)
        
        for action, child in node.children.items():
            # For this implementation, we assume uniform prior probabilities from the diffusion model.
            # If the Diffusion Prior outputs specific action logits, they would be injected here.
            prior = 1.0 / NUM_ACTIONS 
            
            q_val = child.q_value
            u_val = self.c_puct * prior * sqrt_parent_visits / (1 + child.visits)
            
            score = q_val + u_val
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child

    def _expand(self, node: MCTSNode):
        """
        Expands the leaf node using the internal Python physics simulator.
        Generates children for all valid actions.
        """
        valid_actions = self.simulator.get_valid_actions(node.state)
        
        for action in valid_actions:
            # Advance the Python simulator one tick
            next_state = self.simulator.step(node.state, action)
            
            child_node = MCTSNode(state=next_state, parent=node, action_taken=action)
            child_node.is_terminal = self.simulator.is_terminal(next_state)
            
            node.children[action] = child_node
            
        node.is_expanded = True

    def _backpropagate(self, node: MCTSNode, value: float):
        """
        Propagates the evaluated value up the tree to the root.
        """
        current = node
        while current is not None:
            current.visits += 1
            current.value_sum += value
            current = current.parent
            
            # In adversarial games, value flips (value = -value). 
            # ARC-AGI-3 is a single-agent puzzle, so the value remains absolute.