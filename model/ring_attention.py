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


def ring_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    process_group: Optional[dist.ProcessGroup] = None,
    n_rep: int = 1,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Ring Attention forward pass with online softmax.

    Implements memory-efficient attention with sequence parallelism using
    ring topology for K,V communication. Uses online softmax for numerical
    stability when accumulating attention across chunks.

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
    K_j = K.clone()
    V_j = V.clone()
    K_buf = torch.empty_like(K)
    V_buf = torch.empty_like(V)

    for step in range(world_size):
        # Determine which chunk we're processing
        # source_rank tells us which device's original K,V we have
        source_rank = (rank - step) % world_size

        # Start async communication (except last step)
        if step < world_size - 1:
            next_rank = (rank + 1) % world_size
            prev_rank = (rank - 1) % world_size

            send_ops = [
                dist.isend(K_j.contiguous(), next_rank, group=process_group),
                dist.isend(V_j.contiguous(), next_rank, group=process_group),
            ]
            recv_ops = [
                dist.irecv(K_buf, prev_rank, group=process_group),
                dist.irecv(V_buf, prev_rank, group=process_group),
            ]

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
        if step < world_size - 1:
            for op in send_ops + recv_ops:
                op.wait()
            # Swap buffers for next iteration
            K_j, K_buf = K_buf, K_j
            V_j, V_buf = V_buf, V_j

    return O_i


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
