"""
Ring Attention implementation for context parallelism.

Enables distributed attention over long sequences by splitting the sequence
across devices in a ring topology. Each device holds local Q and rotates K,V
around the ring, computing blockwise attention with online softmax.
"""
import math
import torch
import torch.distributed as dist
from typing import Optional, Tuple

# Try to import flash_attn for memory-efficient attention.
# We use _flash_attn_forward (internal API) because the public flash_attn_func
# only exposes softmax_lse when return_attn_probs=True, which also materializes
# the full N^2 attention matrix. _flash_attn_forward with return_softmax=False
# gives us LSE for online softmax accumulation without the memory cost.
# Note: Internal API may change between flash-attn versions.
try:
    from flash_attn.flash_attn_interface import _flash_attn_forward
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand KV heads to match query heads for GQA.

    Args:
        x: [B, n_kv_heads, S, head_dim]
        n_rep: Number of times to repeat each KV head

    Returns:
        [B, n_kv_heads * n_rep, S, head_dim]
    """
    if n_rep == 1:
        return x
    B, n_kv_heads, S, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv_heads, n_rep, S, head_dim)
        .reshape(B, n_kv_heads * n_rep, S, head_dim)
    )


def apply_chunk_causal_mask(
    scores: torch.Tensor,
    local_rank: int,
    source_rank: int,
) -> torch.Tensor:
    """
    Apply causal mask based on chunk positions.

    For causal attention, position i can only attend to positions j <= i.
    When sequences are split across ranks:
    - If source_rank > local_rank: K chunk is entirely in the future -> mask all
    - If source_rank < local_rank: K chunk is entirely in the past -> no mask
    - If source_rank == local_rank: diagonal block -> standard causal mask

    Args:
        scores: [B, n_heads, S_local, S_local] attention scores
        local_rank: Rank holding Q (determines query positions)
        source_rank: Original rank of K,V (determines key positions)

    Returns:
        Masked scores tensor
    """
    if source_rank > local_rank:
        # K chunk is entirely in the future - mask everything
        scores = scores.masked_fill(
            torch.ones_like(scores, dtype=torch.bool),
            float('-inf')
        )
    elif source_rank == local_rank:
        # Diagonal block - standard causal mask (upper triangular)
        S = scores.shape[-1]
        mask = torch.triu(
            torch.ones(S, S, device=scores.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(mask, float('-inf'))
    # else: source_rank < local_rank - K chunk is in the past, no masking

    return scores


def ring_attention_forward_flash(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    process_group: Optional[dist.ProcessGroup] = None,
    n_rep: int = 1,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Ring Attention forward pass using Flash Attention.

    Memory-efficient implementation using flash_attn which provides O(S) memory
    instead of O(S^2) per chunk, plus returns LSE for proper online accumulation.

    Args:
        Q: [B, n_heads, S_local, head_dim] - query (stays on this device)
        K: [B, n_kv_heads, S_local, head_dim] - key (rotates around ring)
        V: [B, n_kv_heads, S_local, head_dim] - value (rotates around ring)
        process_group: ProcessGroup for context parallelism (default: WORLD)
        n_rep: GQA repetition factor (n_heads // n_kv_heads)
        is_causal: Whether to apply causal masking

    Returns:
        O: [B, n_heads, S_local, head_dim] - attention output
    """
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)

    B, n_heads, S_local, head_dim = Q.shape
    n_kv_heads = K.shape[1]

    # flash_attn expects [B, S, n_heads, head_dim] layout
    Q_flash = Q.transpose(1, 2).contiguous()  # [B, S_local, n_heads, head_dim]
    K_j = K.transpose(1, 2).contiguous()      # [B, S_local, n_kv_heads, head_dim]
    V_j = V.transpose(1, 2).contiguous()      # [B, S_local, n_kv_heads, head_dim]

    # Working buffers for K, V rotation
    K_buf = torch.empty_like(K_j)
    V_buf = torch.empty_like(V_j)

    # Initialize accumulators for online softmax
    # O_acc: accumulated weighted output
    # lse_acc: accumulated log-sum-exp (log of sum of attention weights)
    O_acc = None
    lse_acc = None  # [B, n_heads, S_local]

    for step in range(world_size):
        # Determine which chunk we're processing
        source_rank = (rank - step) % world_size

        # Start async communication (except last step)
        reqs = None
        if step < world_size - 1:
            next_rank = (rank + 1) % world_size
            prev_rank = (rank - 1) % world_size

            ops = [
                dist.P2POp(dist.isend, K_j, next_rank, group=process_group),
                dist.P2POp(dist.isend, V_j, next_rank, group=process_group),
                dist.P2POp(dist.irecv, K_buf, prev_rank, group=process_group),
                dist.P2POp(dist.irecv, V_buf, prev_rank, group=process_group),
            ]
            reqs = dist.batch_isend_irecv(ops)

        # Determine causal mode for this chunk pair
        if is_causal and source_rank > rank:
            # K chunk is entirely in the future - skip computation
            # This chunk contributes nothing to attention
            pass
        else:
            # Compute attention for this chunk using flash_attn
            # causal=True only for diagonal block (source_rank == rank)
            use_causal = is_causal and (source_rank == rank)

            # Use _flash_attn_forward to get softmax_lse without materializing
            # the N^2 attention matrix. return_softmax=False is key here.
            out_j, _, _, _, _, lse_j, _, _ = _flash_attn_forward(
                Q_flash, K_j, V_j,
                dropout_p=0.0,
                softmax_scale=None,  # defaults to 1/sqrt(head_dim)
                causal=use_causal,
                window_size=(-1, -1),  # no sliding window
                softcap=0.0,
                alibi_slopes=None,
                return_softmax=False,  # critical: don't materialize N^2 matrix
            )
            # out_j: [B, S_local, n_heads, head_dim]
            # lse_j: [B, n_heads, S_local]

            if O_acc is None:
                # First valid chunk
                O_acc = out_j
                lse_acc = lse_j
            else:
                # Online softmax accumulation using LSE values
                # new_lse = logsumexp(lse_acc, lse_j)
                # O_new = (exp(lse_acc - new_lse) * O_acc + exp(lse_j - new_lse) * out_j)

                lse_max = torch.maximum(lse_acc, lse_j)
                exp_acc = torch.exp(lse_acc - lse_max)
                exp_j = torch.exp(lse_j - lse_max)

                # Sum of exponentials for new normalization
                exp_sum = exp_acc + exp_j

                # Weighted combination of outputs
                # Reshape for broadcasting: [B, n_heads, S_local] -> [B, S_local, n_heads, 1]
                exp_acc_b = exp_acc.transpose(1, 2).unsqueeze(-1)
                exp_j_b = exp_j.transpose(1, 2).unsqueeze(-1)
                exp_sum_b = exp_sum.transpose(1, 2).unsqueeze(-1)

                O_acc = (exp_acc_b * O_acc + exp_j_b * out_j) / exp_sum_b

                # Update LSE accumulator
                lse_acc = lse_max + torch.log(exp_sum)

        # Wait for communication to complete
        if reqs is not None:
            for req in reqs:
                req.wait()
            K_j, K_buf = K_buf, K_j
            V_j, V_buf = V_buf, V_j

    # Handle edge case where all chunks were masked (shouldn't happen in practice)
    if O_acc is None:
        O_acc = torch.zeros_like(Q_flash)

    # Convert back to [B, n_heads, S_local, head_dim] layout
    return O_acc.transpose(1, 2).contiguous()


def ring_attention_forward_eager(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    process_group: Optional[dist.ProcessGroup] = None,
    n_rep: int = 1,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Ring Attention forward pass with online softmax (eager mode fallback).

    Implements memory-efficient attention with sequence parallelism using
    ring topology for K,V communication. Uses online softmax for numerical
    stability when accumulating attention across chunks.

    Note: This implementation materializes O(S^2) attention scores per chunk.
    Use ring_attention_forward_flash when flash_attn is available for better
    memory efficiency.

    Args:
        Q: [B, n_heads, S_local, head_dim] - query (stays on this device)
        K: [B, n_kv_heads, S_local, head_dim] - key (rotates around ring)
        V: [B, n_kv_heads, S_local, head_dim] - value (rotates around ring)
        process_group: ProcessGroup for context parallelism (default: WORLD)
        n_rep: GQA repetition factor (n_heads // n_kv_heads)
        is_causal: Whether to apply causal masking

    Returns:
        O: [B, n_heads, S_local, head_dim] - attention output
    """
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    print(f"Ring attention forward on rank {rank} with world size {world_size}")

    B, n_heads, S_local, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)

    # Initialize online softmax accumulators
    # m_i: running max for numerical stability
    # l_i: running sum of exp(scores - m_i) for normalization
    # O_i: running weighted sum of values
    m_i = torch.full(
        (B, n_heads, S_local, 1),
        float('-inf'),
        device=Q.device,
        dtype=Q.dtype
    )
    l_i = torch.zeros(
        (B, n_heads, S_local, 1),
        device=Q.device,
        dtype=Q.dtype
    )
    O_i = torch.zeros_like(Q)

    # Working buffers for K, V rotation
    K_j = K.clone().contiguous()
    V_j = V.clone().contiguous()        
    K_buf = torch.empty_like(K).contiguous()
    V_buf = torch.empty_like(V).contiguous()
    print(f"Ring attention forward on rank {rank} with world size {world_size} initialized buffers")
    for step in range(world_size):
        # Determine which chunk we're processing
        # source_rank tells us which device's original K,V we have
        source_rank = (rank - step) % world_size

        # Start async communication (except last step)
        reqs = None                                                                
        if step < world_size - 1:                                                  
            next_rank = (rank + 1) % world_size                                    
            prev_rank = (rank - 1) % world_size                                    
                                                                                     
            ops = [                                                                
                dist.P2POp(dist.isend, K_j, next_rank, group=process_group),       
                dist.P2POp(dist.isend, V_j, next_rank, group=process_group),       
                dist.P2POp(dist.irecv, K_buf, prev_rank, group=process_group),     
                dist.P2POp(dist.irecv, V_buf, prev_rank, group=process_group),     
            ]
                                                                                  
            reqs = dist.batch_isend_irecv(ops)
            print(f"Ring attention forward on rank {rank} with world size {world_size} sent and received data")
        # Expand K, V for GQA
        K_expanded = repeat_kv(K_j, n_rep)  # [B, n_heads, S_local, head_dim]
        V_expanded = repeat_kv(V_j, n_rep)

        # Compute attention scores: Q @ K^T * scale
        scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) * scale
        # scores: [B, n_heads, S_local, S_local]

        # Apply causal mask for this chunk pair
        if is_causal:
            scores = apply_chunk_causal_mask(scores, rank, source_rank)

        # --- Online softmax update (Flash Attention style) ---

        # Get max of current block for numerical stability
        m_ij = scores.amax(dim=-1, keepdim=True)
        # Handle all -inf case (fully masked block)
        m_ij = torch.where(
            m_ij == float('-inf'),
            torch.zeros_like(m_ij),
            m_ij
        )

        # New running max
        m_new = torch.maximum(m_i, m_ij)

        # Correction factors for rescaling
        # alpha: rescale factor for old accumulator
        # beta: scale factor for current block
        alpha = torch.exp(
            torch.where(
                m_i == float('-inf'),
                torch.zeros_like(m_i),
                m_i - m_new
            )
        )
        beta = torch.exp(m_ij - m_new)

        # Softmax of current block (unnormalized)
        P_ij = torch.exp(scores - m_ij)
        # Zero out masked positions
        P_ij = torch.where(
            scores == float('-inf'),
            torch.zeros_like(P_ij),
            P_ij
        )

        # Update running sum
        l_new = alpha * l_i + beta * P_ij.sum(dim=-1, keepdim=True)

        # Update output accumulator
        # O_new = (alpha * l_i * O_i + beta * P_ij @ V_j) / l_new
        # Avoid division by zero for fully masked sequences
        l_safe = torch.where(l_new == 0, torch.ones_like(l_new), l_new)
        O_i = (alpha * l_i * O_i + beta * torch.matmul(P_ij, V_expanded)) / l_safe

        # Store new running values
        m_i = m_new
        l_i = l_new

        # Wait for communication to complete   
        print(f"Ring attention forward on rank {rank} with world size {world_size} waiting for communication to complete")
        if reqs is not None:                                                       
            for req in reqs:                                                       
                req.wait()                                                         
            # Swap buffers for next iteration                                      
            K_j, K_buf = K_buf, K_j                                                
            V_j, V_buf = V_buf, V_j
        print(f"Ring attention forward on rank {rank} with world size {world_size} communication completed")

    return O_i


def ring_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    process_group: Optional[dist.ProcessGroup] = None,
    n_rep: int = 1,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Ring Attention forward pass - dispatches to flash or eager implementation.

    Uses flash_attn when available for O(S) memory per chunk.
    Falls back to eager implementation with O(S^2) memory otherwise.

    Args:
        Q: [B, n_heads, S_local, head_dim] - query (stays on this device)
        K: [B, n_kv_heads, S_local, head_dim] - key (rotates around ring)
        V: [B, n_kv_heads, S_local, head_dim] - value (rotates around ring)
        process_group: ProcessGroup for context parallelism (default: WORLD)
        n_rep: GQA repetition factor (n_heads // n_kv_heads)
        is_causal: Whether to apply causal masking

    Returns:
        O: [B, n_heads, S_local, head_dim] - attention output
    """
    if HAS_FLASH_ATTN and Q.is_cuda:
        return ring_attention_forward_flash(Q, K, V, process_group, n_rep, is_causal)
    else:
        return ring_attention_forward_eager(Q, K, V, process_group, n_rep, is_causal)


def ring_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    process_group: Optional[dist.ProcessGroup] = None,
    n_rep: int = 1,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Ring Attention - main entry point.

    Currently only supports forward pass. For training with gradients,
    use RingAttentionFunc (torch.autograd.Function) - to be implemented.

    Uses flash_attn when available for memory-efficient attention (O(S) vs O(S^2)).

    Args:
        Q: [B, n_heads, S_local, head_dim]
        K: [B, n_kv_heads, S_local, head_dim]
        V: [B, n_kv_heads, S_local, head_dim]
        process_group: ProcessGroup for context parallelism
        n_rep: GQA repetition factor
        is_causal: Whether to apply causal masking

    Returns:
        O: [B, n_heads, S_local, head_dim]
    """
    return ring_attention_forward(Q, K, V, process_group, n_rep, is_causal)
