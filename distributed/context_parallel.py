"""
Context parallelism utilities for distributing sequences across devices.
"""
import torch
import torch.distributed as dist
from typing import Optional, List


class ContextParallelManager:
    """
    Manages context parallelism state and provides utilities for
    splitting/gathering sequences across devices.
    """

    def __init__(self, process_group: Optional[dist.ProcessGroup] = None):
        self.group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)

    def split_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split sequence tensor along sequence dimension.

        Args:
            x: [B, S_global, ...] tensor (same on all ranks)

        Returns:
            [B, S_local, ...] with this rank's chunk
        """
        B, S_global = x.shape[:2]
        assert S_global % self.world_size == 0, \
            f"Sequence length {S_global} not divisible by {self.world_size}"

        S_local = S_global // self.world_size
        start = self.rank * S_local
        end = start + S_local

        return x[:, start:end].contiguous()

    def gather_sequence(self, x_local: torch.Tensor) -> torch.Tensor:
        """
        Gather sequence chunks from all ranks.

        Args:
            x_local: [B, S_local, ...] with this rank's chunk

        Returns:
            [B, S_global, ...] with full sequence on all ranks
        """
        gathered: List[torch.Tensor] = [
            torch.empty_like(x_local) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered, x_local.contiguous(), group=self.group)

        return torch.cat(gathered, dim=1)

    def get_position_offset(self, local_seq_len: int) -> int:
        """Get global position offset for RoPE."""
        return self.rank * local_seq_len

    def prepare_rope_freqs(
        self,
        freqs_cis_full: torch.Tensor,
        local_seq_len: int
    ) -> torch.Tensor:
        """
        Extract RoPE frequencies for this rank's sequence positions.

        Args:
            freqs_cis_full: Full RoPE frequencies [max_seq_len, head_dim/2]
            local_seq_len: Length of local sequence chunk

        Returns:
            RoPE frequencies for local positions [local_seq_len, head_dim/2]
        """
        offset = self.get_position_offset(local_seq_len)
        return freqs_cis_full[offset : offset + local_seq_len]
