import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple


class GridSequencePacker:
    """
    A utility class to pack variable-length 2D ARC grids into a contiguous 1D buffer.

    This enables the use of FlashAttention / xFormers memory-efficient attention
    without wasting FLOPs on <PAD> tokens. It also extracts the 2D spatial
    coordinates (row, col) necessary for applying 2D RoPE before attention.
    """

    @staticmethod
    def pack_grids(
        grids: List[torch.Tensor], device: torch.device | None = None
    ) -> Dict[str, Any]:
        """
        Takes a list of 2D or flattened grid tensors and packs them into a 1D sequence.

        Args:
            grids: List of tensors. Can be raw integer grids [H, W] or embedded
                   token sequences [H * W, embed_dim].
            device: Target torch device.

        Returns:
            A dictionary containing:
                - 'packed_sequence': The concatenated 1D tensor.
                - 'cu_seq_lens': Cumulative lengths array [batch_size + 1] (int32).
                - 'max_seq_len': The maximum sequence length in the batch.
                - 'grid_shapes': List of original shapes for unpacking/RoPE.
                - 'row_coords': 1D tensor of row indices for each token.
                - 'col_coords': 1D tensor of column indices for each token.
        """
        if not grids:
            raise ValueError("Cannot pack an empty grid list")

        if device is None:
            device = grids[0].device

        packed_tensors = []
        lengths = []
        grid_shapes: List[Tuple[int, ...]] = []
        row_coords_list = []
        col_coords_list = []

        for grid in grids:
            grid = grid.to(device)
            # Handle both raw grids [H, W] and embedded sequence grids [H, W, D]
            if grid.dim() == 1:
                seq_len = grid.shape[0]
                if seq_len == 0:
                    raise ValueError("Cannot pack an empty 1D sequence")
                flattened = grid.contiguous()
                grid_shapes.append((seq_len,))
                lengths.append(seq_len)
                packed_tensors.append(flattened)
                row_coords_list.append(
                    torch.zeros(seq_len, dtype=torch.long, device=device)
                )
                col_coords_list.append(
                    torch.arange(seq_len, dtype=torch.long, device=device)
                )
                continue

            if grid.dim() == 2:
                h, w = grid.shape
                flattened = grid.contiguous().view(-1)
            elif grid.dim() == 3:
                h, w, d = grid.shape
                flattened = grid.contiguous().view(-1, d)
            else:
                raise ValueError(f"Expected 1D, 2D, or 3D grid, got shape {grid.shape}")
            if h <= 0 or w <= 0:
                raise ValueError(f"Invalid grid shape {(h, w)}")

            grid_shapes.append((h, w))
            seq_len = h * w
            lengths.append(seq_len)
            packed_tensors.append(flattened)

            # Generate 2D spatial coordinates for Unsloth 2D RoPE
            rows, cols = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            row_coords_list.append(rows.flatten())
            col_coords_list.append(cols.flatten())

        # 1. Concatenate into a single contiguous buffer
        packed_sequence = torch.cat(packed_tensors, dim=0)

        # 2. Build cumulative sequence lengths (Must be int32 for xFormers)
        lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=device)
        cu_seq_lens = torch.zeros(len(grids) + 1, dtype=torch.int32, device=device)
        cu_seq_lens[1:] = torch.cumsum(lengths_tensor, dim=0)

        # 3. Concatenate coordinate maps
        row_coords = torch.cat(row_coords_list, dim=0)
        col_coords = torch.cat(col_coords_list, dim=0)

        return {
            "packed_sequence": packed_sequence,
            "cu_seq_lens": cu_seq_lens,
            "max_seq_len": int(lengths_tensor.max().item()),
            "grid_shapes": grid_shapes,
            "row_coords": row_coords,
            "col_coords": col_coords,
        }

    @staticmethod
    def unpack_sequence(
        packed_sequence: torch.Tensor, grid_shapes: List[Tuple[int, ...]]
    ) -> List[torch.Tensor]:
        """
        Reverses the packing process, returning a list of 2D/3D grids.
        Useful when translating continuous token probabilities back into a grid output.
        """
        unpacked_grids = []
        current_idx = 0

        for shape in grid_shapes:
            if len(shape) == 1:
                seq_len = shape[0]
            elif len(shape) == 2:
                h, w = shape
                seq_len = h * w
            else:
                h, w, _ = shape
                seq_len = h * w
            # Extract sequence slice
            seq_slice = packed_sequence[current_idx : current_idx + seq_len]
            current_idx += seq_len

            # Reshape back to original 2D or 3D geometry
            if len(shape) == 1:
                unpacked_grids.append(seq_slice.view(shape[0]))
            elif seq_slice.dim() == 1:
                h, w = shape
                unpacked_grids.append(seq_slice.view(h, w))
            else:
                # Assuming [seq_len, embed_dim]
                h, w = shape[:2]
                d = seq_slice.shape[-1]
                unpacked_grids.append(seq_slice.view(h, w, d))

        return unpacked_grids


# ==========================================
# Memory-Efficient Attention Wrapper
# ==========================================


def execute_packed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """
    Executes scaled dot-product attention on packed 1D sequences.

    Args:
        query, key, value: Tensors of shape [total_tokens, num_heads, head_dim].
        cu_seq_lens: int32 tensor of shape [batch_size + 1] outlining sequence boundaries.
        max_seq_len: The maximum single sequence length in the packed batch.

    Returns:
        attn_output: Tensor of shape [total_tokens, num_heads, head_dim].
    """

    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical packed shapes")
    if query.dim() != 3:
        raise ValueError(
            "Packed attention expects [total_tokens, num_heads, head_dim] tensors"
        )
    if cu_seq_lens.dim() != 1 or cu_seq_lens.numel() < 2:
        raise ValueError("cu_seq_lens must be a 1D tensor with at least two entries")

    lengths = (cu_seq_lens[1:] - cu_seq_lens[:-1]).to(torch.long)
    if int(lengths.max().item()) != int(max_seq_len):
        raise ValueError("max_seq_len does not match cu_seq_lens")

    if query.is_cuda:
        try:
            import xformers.ops as xops

            attn_bias = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                lengths.tolist()
            )
            output = xops.memory_efficient_attention(
                query.unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                attn_bias=attn_bias,
            )
            return output.squeeze(0)
        except Exception:
            # Fall back to the exact PyTorch implementation below. This keeps the
            # code usable on offline Kaggle images missing a compatible xFormers wheel.
            pass

    outputs = []
    for start, end in zip(cu_seq_lens[:-1].tolist(), cu_seq_lens[1:].tolist()):
        q_i = query[start:end].permute(1, 0, 2).unsqueeze(0)
        k_i = key[start:end].permute(1, 0, 2).unsqueeze(0)
        v_i = value[start:end].permute(1, 0, 2).unsqueeze(0)
        out_i = F.scaled_dot_product_attention(
            q_i, k_i, v_i, dropout_p=0.0, is_causal=False
        )
        outputs.append(out_i.squeeze(0).permute(1, 0, 2))

    return torch.cat(outputs, dim=0)
