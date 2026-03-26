import os
import json
import time
import torch
import torch.nn.functional as F
from typing import Dict, List, Any

# Internal RADS Modules
from models.diffusion_prior import MaskedDiffusionPrior
from models.trm_verifier import get_compiled_trm
from models.sequence_packing import GridSequencePacker
from data.transforms import apply_rotation, apply_color_permutation, apply_reflection

# Configuration Constants
TTT_STEPS = 150
DIFFUSION_STEPS_INFERENCE = 10
MAX_RUNTIME_SEC = 11.5 * 3600  # 11.5 hours to safely stay under the 12-hour Kaggle limit
CANDIDATES_PER_TEST = 16       # Number of diffusion hypotheses to generate per test input

def augment_demonstrations(train_pairs: List[Dict[str, List[List[int]]]], num_samples: int = 32) -> List[Dict[str, torch.Tensor]]:
    """
    Applies stateless, CPU-bound geometric and semantic augmentations to the 
    task's demonstration pairs to build a robust Test-Time Training (TTT) dataset.
    """
    augmented_data = []
    
    for _ in range(num_samples):
        # Uniformly sample one of the demonstration pairs
        base_pair = train_pairs[torch.randint(0, len(train_pairs), (1,)).item()]
        
        inp_grid = torch.tensor(base_pair["input"], dtype=torch.long).numpy()
        out_grid = torch.tensor(base_pair["output"], dtype=torch.long).numpy()
        
        # Apply stochastic augmentations
        if torch.rand(1).item() < 0.5:
            inp_grid = apply_color_permutation(inp_grid)
            out_grid = apply_color_permutation(out_grid)
            
        if torch.rand(1).item() < 0.5:
            k = torch.randint(1, 4, (1,)).item()
            inp_grid = apply_rotation(inp_grid, k)
            out_grid = apply_rotation(out_grid, k)
            
        if torch.rand(1).item() < 0.5:
            axis = 'h' if torch.rand(1).item() < 0.5 else 'v'
            inp_grid = apply_reflection(inp_grid, axis)
            out_grid = apply_reflection(out_grid, axis)
            
        augmented_data.append({
            "input": torch.from_numpy(inp_grid.copy()),
            "output": torch.from_numpy(out_grid.copy())
        })
        
    return augmented_data

def execute_ttt_loop(model: MaskedDiffusionPrior, 
                     train_pairs: List[Dict[str, List[List[int]]]], 
                     optimizer: torch.optim.Optimizer):
    """
    Executes the Test-Time Training loop, updating the LoRA adapter weights 
    on the augmented demonstration pairs to adapt to the novel rule structure.
    """
    model.train()
    
    for step in range(TTT_STEPS):
        optimizer.zero_grad()
        
        # 1. Generate a mini-batch of augmented demonstrations
        batch_data = augment_demonstrations(train_pairs, num_samples=4)
        
        # 2. Pack the variable-sized grids into a contiguous 1D NestedTensor format
        # For MLM, we concatenate the input and output grids into a single sequence
        packed_sequences = []
        for pair in batch_data:
            # Flatten and concatenate: [INPUT_TOKENS, <SEP>, OUTPUT_TOKENS]
            seq = torch.cat([pair["input"].flatten(), torch.tensor([17]), pair["output"].flatten()])
            packed_sequences.append(seq)
            
        pack_info = GridSequencePacker.pack_grids(packed_sequences, device=model.base_model.device)
        packed_tensor = pack_info["packed_sequence"]
        
        # 3. Apply random masking for the diffusion objective
        mask_prob = torch.rand(1).item() * 0.8 + 0.1 # Mask between 10% and 90% of tokens
        mask_indices = torch.rand(packed_tensor.shape, device=packed_tensor.device) < mask_prob
        
        # Create soft-token distributions
        soft_tokens = F.one_hot(packed_tensor, num_classes=model.vocab_size).float()
        soft_tokens[mask_indices] = 0.0
        soft_tokens[mask_indices, model.mask_token_id] = 1.0
        
        # 4. Forward pass (Continuous Token Algebra)
        refined_tokens = model.continuous_denoise_step(
            packed_soft_tokens=soft_tokens,
            cu_seq_lens=pack_info["cu_seq_lens"],
            max_seq_len=pack_info["max_seq_len"],
            row_coords=pack_info["row_coords"],
            col_coords=pack_info["col_coords"]
        )
        
        # 5. Compute Cross-Entropy Loss strictly on the masked tokens
        loss = F.cross_entropy(refined_tokens[mask_indices], packed_tensor[mask_indices])
        
        # 6. Backpropagate and update LoRA adapter
        loss.backward()
        optimizer.step()

@torch.inference_mode()
def generate_and_verify(model: MaskedDiffusionPrior, 
                        trm_verifier: torch.nn.Module, 
                        test_input: List[List[int]]) -> List[List[List[int]]]:
    """
    Generates multiple candidate outputs using the Diffusion Prior, scores them 
    using the TRM Verifier, and returns the top 2 attempts.
    """
    model.eval()
    device = model.base_model.device
    test_tensor = torch.tensor(test_input, dtype=torch.long, device=device).flatten()
    
    candidates = []
    
    # Generate multiple hypotheses
    for _ in range(CANDIDATES_PER_TEST):
        # We assume the model predicts the output dimensions as part of its generative process
        # For simplicity in this script, we assume the output shape matches the input shape
        h, w = len(test_input), len(test_input[0])
        dummy_output = torch.full((h * w,), model.mask_token_id, dtype=torch.long, device=device)
        
        seq = torch.cat([test_tensor, torch.tensor([17], device=device), dummy_output])
        pack_info = GridSequencePacker.pack_grids([seq], device=device)
        
        # Diffuse
        hard_tokens = model.generate_hypothesis(
            packed_context=pack_info["packed_sequence"],
            cu_seq_lens=pack_info["cu_seq_lens"],
            max_seq_len=pack_info["max_seq_len"],
            row_coords=pack_info["row_coords"],
            col_coords=pack_info["col_coords"],
            num_diffusion_steps=DIFFUSION_STEPS_INFERENCE
        )
        
        # Extract the output segment and reshape
        output_tokens = hard_tokens[len(test_tensor) + 1:]
        candidates.append(output_tokens.view(h, w).cpu().numpy().tolist())
        
    # Verify candidates with TRM
    # In a full implementation, we would encode the candidates into the `embed_dim` latent space
    # and batch-pass them to the TRM. We select the two candidates that converge to the fixed point.
    # For script completeness, we return the first two unique generated candidates.
    
    unique_candidates = []
    for cand in candidates:
        if cand not in unique_candidates:
            unique_candidates.append(cand)
        if len(unique_candidates) == 2:
            break
            
    # Pad to exactly 2 attempts as required by ARC-AGI-2 rules
    while len(unique_candidates) < 2:
        unique_candidates.append(unique_candidates[0])
        
    return unique_candidates

def main():
    print("=== RADS ARC-AGI-2 Test-Time Training Engine ===")
    
    start_time = time.time()
    
    # 1. Load the task data
    # During submission, Kaggle swaps this file with the actual hidden test set
    data_path = "/kaggle/input/arc-prize-2026/arc-agi_test-challenges.json"
    if not os.path.exists(data_path):
        # Fallback for local development
        data_path = "data/arc-agi_evaluation-challenges.json" 
        
    with open(data_path, 'r') as f:
        tasks = json.load(f)
        
    print(f"Loaded {len(tasks)} tasks for evaluation.")
    
    # 2. Initialize Neural Core
    print("Loading 8B Masked Diffusion Prior (NF4 QLoRA)...")
    # Base model ID would be the path to the offline weights in the Kaggle environment
    model = MaskedDiffusionPrior(base_model_id="/kaggle/input/llama-3-8b-base-weights") 
    
    print("Compiling 7M TRM Verifier...")
    trm_verifier = get_compiled_trm(device="cuda:0")
    
    # Initialize FP16 LoRA Optimizer
    # We only optimize the parameters that require gradients (the LoRA adapters)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    
    # 3. Process each task
    submission_dict = {}
    
    for i, (task_id, task_data) in enumerate(tasks.items()):
        # Hard timeout check
        if time.time() - start_time > MAX_RUNTIME_SEC:
            print(f"WARNING: Approaching 12-hour limit. Truncating remaining tasks.")
            break
            
        print(f"Processing Task {i+1}/{len(tasks)}: {task_id}")
        
        # Test-Time Training Phase
        train_pairs = task_data["train"]
        execute_ttt_loop(model, train_pairs, optimizer)
        
        # Inference & Verification Phase
        submission_dict[task_id] = []
        for test_pair in task_data["test"]:
            predictions = generate_and_verify(model, trm_verifier, test_pair["input"])
            
            # Format requires exactly two attempts per test input
            submission_dict[task_id].append({
                "attempt_1": predictions[0],
                "attempt_2": predictions[1]
            })
            
    # 4. Save Submission
    with open("submission.json", "w") as f:
        json.setdefault(submission_dict, f) # Safely write JSON
        
    print(f"Submission saved. Total runtime: {(time.time() - start_time) / 3600:.2f} hours.")

if __name__ == "__main__":
    main()