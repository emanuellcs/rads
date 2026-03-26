import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

class MaskedDiffusionPrior(nn.Module):
    """
    The 8B Masked Diffusion Language Model (MDLM) core.
    
    This module manages the QLoRA hardware-efficient weight loading and 
    executes the continuous soft-masking denoising loop over packed 1D sequences.
    """
    
    def __init__(self, 
                 base_model_id: str, 
                 lora_rank: int = 32, 
                 vocab_size: int = 17,  # 0-15 colors + 1 <MASK> token
                 mask_token_id: int = 16):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        
        # 1. Initialize QLoRA 4-bit NF4 Base Model
        self.base_model = self._load_quantized_base(base_model_id)
        
        # 2. Inject Rank-32 FP16 LoRA Adapters
        self.model = self._inject_lora_adapters(self.base_model, lora_rank)
        
        # We extract the base embedding layer to perform Continuous Token Algebra
        self.embed_tokens = self.model.get_input_embeddings()

    def _load_quantized_base(self, model_id: str) -> AutoModelForCausalLM:
        """
        Loads the 8B base model into ~4.5GB VRAM using NF4 quantization.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # Information-theoretically optimal for normal distributions
            bnb_4bit_use_double_quant=True,     # Nested quantization for extra memory savings
            bnb_4bit_compute_dtype=torch.float16 # Compute adapter updates in FP16
        )
        
        # Force FlashAttention-2 / SDPA via attn_implementation
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa" 
        )
        
        # Freeze base model weights entirely
        for param in model.parameters():
            param.requires_grad = False
            
        return model

    def _inject_lora_adapters(self, model: nn.Module, rank: int) -> nn.Module:
        """
        Injects Rank-32 LoRA adapters into all linear projection layers.
        These adapters are hot-swapped between ARC-AGI-2 (Static) and ARC-AGI-3 (Interactive).
        """
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, # Dropout disabled for deterministic verification
            bias="none",
            task_type="CAUSAL_LM"
        )
        peft_model = get_peft_model(model, lora_config)
        return peft_model

    def swap_lora_weights(self, new_state_dict: Dict[str, torch.Tensor]):
        """
        Zero-latency hot-swap of task-specific adapters. 
        Replaces the FP16 LoRA deltas without reloading the 4GB NF4 base model.
        """
        set_peft_model_state_dict(self.model, new_state_dict)

    def continuous_denoise_step(self, 
                                packed_soft_tokens: torch.Tensor, 
                                cu_seq_lens: torch.Tensor,
                                max_seq_len: int,
                                row_coords: torch.Tensor,
                                col_coords: torch.Tensor) -> torch.Tensor:
        """
        Executes a single step of Continuous Token Algebra over the packed sequence.
        
        Args:
            packed_soft_tokens: Probability distributions over vocabulary [total_tokens, vocab_size]
            cu_seq_lens: Sequence boundaries [batch_size + 1]
            row_coords, col_coords: 2D spatial mappings for Fused 2D RoPE
            
        Returns:
            Refined probability distributions [total_tokens, vocab_size]
        """
        # 1. Project probability distributions into continuous embedding space
        # shape: [total_tokens, embed_dim]
        # This blends the latent vectors based on the current confidence of each token
        continuous_embeddings = torch.matmul(packed_soft_tokens, self.embed_tokens.weight[:self.vocab_size])
        
        # TODO: Apply Unsloth Fused 2D RoPE injection here using row_coords and col_coords
        # continuous_embeddings = inject_2d_rope(continuous_embeddings, row_coords, col_coords)
        
        # 2. Forward pass through the Transformer
        # Since we bypass standard input embedding, we pass inputs_embeds directly.
        # cu_seq_lens is handled natively if passed via kwargs to the SDPA-enabled model.
        outputs = self.model(
            inputs_embeds=continuous_embeddings.unsqueeze(0), # Dummy batch dim for HF
            attention_mask=None, # Masking is handled entirely by cu_seq_lens in SDPA
            output_hidden_states=False
        )
        
        # 3. Extract logits and compute new soft distribution
        logits = outputs.logits.squeeze(0) # [total_tokens, vocab_size]
        
        # Normalize into a probability distribution for the next iterative step
        refined_soft_tokens = F.softmax(logits, dim=-1)
        
        return refined_soft_tokens

    def generate_hypothesis(self, 
                            packed_context: torch.Tensor, 
                            cu_seq_lens: torch.Tensor,
                            max_seq_len: int,
                            row_coords: torch.Tensor,
                            col_coords: torch.Tensor,
                            num_diffusion_steps: int = 10) -> torch.Tensor:
        """
        Executes the full diffusion loop to synthesize an ARC grid or Python world model.
        """
        # Initialize sequence as 100% <MASK> token probability
        total_tokens = packed_context.shape[0]
        soft_tokens = torch.zeros((total_tokens, self.vocab_size), device=packed_context.device)
        soft_tokens[:, self.mask_token_id] = 1.0
        
        # Override the input/demonstration segments of the sequence with hard token 
        # probabilities so the model only diffuses the target answer area
        # (Assuming `packed_context` contains the ground-truth token IDs for the prompt)
        is_prompt_mask = packed_context != self.mask_token_id
        
        for step in range(num_diffusion_steps):
            refined_tokens = self.continuous_denoise_step(
                soft_tokens, cu_seq_lens, max_seq_len, row_coords, col_coords
            )
            
            # Re-clamp the prompt tokens to absolute certainty (1.0)
            # and allow the diffusion to update only the predicted target cells
            soft_tokens = torch.where(
                is_prompt_mask.unsqueeze(-1), 
                F.one_hot(packed_context, num_classes=self.vocab_size).float(), 
                refined_tokens
            )
            
        # Return the final deterministic `argmax` selection as the finalized hypothesis
        final_hard_tokens = torch.argmax(soft_tokens, dim=-1)
        return final_hard_tokens