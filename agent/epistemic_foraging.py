import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
import math

# Abstract action mapping based on typical ARC-AGI-3 environment assumptions
ACTION_RIGHT = 1
ACTION_DOWN = 3
ACTION_TOGGLE = 6
ACTION_RESET = 7

class EpistemicForager:
    """
    Manages the physical exploration phase of the ARC-AGI-3 agent.
    
    Executes Minimum Viable Probes (MVP) to ground spatial coordinates,
    manages the RESET exploit for hazard mapping, and calculates the 
    Homogeneous Pragmatic Consensus (HPC) stopping condition.
    """
    
    def __init__(self, 
                 max_resets: int = 3, 
                 beam_size: int = 16, 
                 attractor_tolerance: float = 0.05):
        """
        Args:
            max_resets: B_RESET bound for the suicide-probing exploit.
            beam_size: Number of Python world-model hypotheses maintained (B=16).
            attractor_tolerance: Delta for TRM fixed-point consensus (delta=0.05).
        """
        self.max_resets = max_resets
        self.beam_size = beam_size
        self.attractor_tolerance = attractor_tolerance
        
        self.resets_used = 0
        self.mvp_completed = False

    def execute_mvp_sequence(self, env_interface) -> List[Dict[str, Any]]:
        """
        Executes the 4-step deterministic Minimum Viable Probe (MVP) sequence.
        This Grounds the X/Y axes, checks toggle states, and probes grid boundaries
        to prevent the Diffusion Prior from hallucinating physical laws.
        
        Args:
            env_interface: The interaction hook to the ARC-AGI-3 environment.
            
        Returns:
            trajectory: A list of (state, action, next_state) transitions.
        """
        trajectory = []
        
        # Helper to step and record
        def _probe(action: int) -> Dict[str, Any]:
            pre_state = env_interface.get_current_frame()
            env_interface.step(action)
            post_state = env_interface.get_current_frame()
            
            transition = {
                "pre_state": pre_state,
                "action": action,
                "post_state": post_state,
                "is_game_over": env_interface.is_game_over()
            }
            trajectory.append(transition)
            return transition

        # Step 1: Axis-Lock Test (X) - Assume ACTION 1 is horizontal
        _probe(ACTION_RIGHT)
        
        # Step 2: Axis-Lock Test (Y) - Assume ACTION 3 is vertical
        _probe(ACTION_DOWN)
        
        # Step 3: Toggle/Interaction Test - Check if clicking self alters cell value
        _probe(ACTION_TOGGLE)
        
        # Step 4: Boundary Probe - Intentionally walk off the known map
        # We extrapolate the opposite of ACTION_RIGHT (e.g., ACTION 0 or 2) 
        # to walk back past the 0-index boundary.
        # For simplicity in this implementation, we simulate moving left out-of-bounds.
        _probe(0) 
        
        # If the boundary probe was lethal, we consume a RESET
        if env_interface.is_game_over():
            env_interface.step(ACTION_RESET)
            self.resets_used += 1
            
        self.mvp_completed = True
        return trajectory

    def deliberate_hazard_probe(self, env_interface, suspected_hazard_action: int) -> bool:
        """
        Executes the 'RESET Exploit' by intentionally suiciding into a suspected trap.
        Because human baselines include panics/resets, the RHAE metric absorbs this penalty.
        """
        if self.resets_used >= self.max_resets:
            return False # Budget exhausted, revert to safe play
            
        env_interface.step(suspected_hazard_action)
        
        if env_interface.is_game_over():
            # Confirm hazard, reset the board
            env_interface.step(ACTION_RESET)
            self.resets_used += 1
            return True
            
        return False

    def check_hpc_condition(self, 
                            active_world_models: List[Any], 
                            predicted_action_sequences: List[List[int]], 
                            trm_latent_states: torch.Tensor) -> bool:
        """
        Evaluates the Homogeneous Pragmatic Consensus (HPC) stopping criterion.
        
        If HPC is met, the agent stops epistemic exploration (which costs RHAE points)
        and immediately executes the winning pragmatic sequence.
        
        Args:
            active_world_models: The remaining Python world-model hypotheses that 
                                 have not been falsified by physical observations.
            predicted_action_sequences: The MCTS winning sequences predicted by each model.
            trm_latent_states: The terminal [embed_dim] vectors from the TRM verifier 
                               for each active model. Shape: [num_active, embed_dim]
                               
        Returns:
            True if exploration should terminate.
        """
        num_active = len(active_world_models)
        
        # 1. Entropy Collapse Check: Must have at least one valid model
        if num_active == 0:
            raise ValueError("All world models falsified. Diffusion Prior must re-sample.")
            
        if num_active == 1:
            # Trivial consensus
            return True

        # 2. Unanimous Action Prediction Check
        # H({a_1^(i), ..., a_m^(i)}) = 0
        reference_sequence = predicted_action_sequences[0]
        for seq in predicted_action_sequences[1:]:
            if seq != reference_sequence:
                # Entropy > 0; the models disagree on the path to victory.
                # The agent must find an action that disambiguates these models.
                return False

        # 3. TRM Attractor Consensus Check
        # max || z_i^* - z_j^* ||_2 < delta
        # Calculate pairwise L2 distances between all converged latent vectors
        # using broadcasting: [num_active, 1, embed_dim] - [1, num_active, embed_dim]
        diffs = trm_latent_states.unsqueeze(1) - trm_latent_states.unsqueeze(0)
        distances = torch.linalg.vector_norm(diffs, ord=2, dim=-1) # [num_active, num_active]
        
        max_distance = torch.max(distances).item()
        
        if max_distance < self.attractor_tolerance:
            return True
            
        return False

def calculate_expected_information_gain(hypotheses: List[Any], candidate_action: int) -> float:
    """
    Calculates the expected reduction in uncertainty (EIG) over the world model hypotheses 
    if a specific action is taken.
    
    This function guides the agent when HPC is NOT met, forcing it to choose the action 
    that maximally shatters the current hypothesis beam.
    """
    # Group hypotheses by the resulting state they predict for this action
    predicted_outcomes = {}
    
    for hyp in hypotheses:
        # Simulate the action in this specific Python hypothesis
        predicted_state_hash = hash(hyp.simulate_step(candidate_action).tobytes())
        
        if predicted_state_hash not in predicted_outcomes:
            predicted_outcomes[predicted_state_hash] = 0
        predicted_outcomes[predicted_state_hash] += 1
        
    # Calculate Shannon Entropy of the partition
    total = len(hypotheses)
    entropy = 0.0
    for count in predicted_outcomes.values():
        p = count / total
        entropy -= p * math.log2(p)
        
    return entropy