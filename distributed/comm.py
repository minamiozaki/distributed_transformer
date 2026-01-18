"""
Ring communication primitives for context parallelism.
"""
import torch
import torch.distributed as dist
from typing import List, Optional


class RingCommunicator:
    """Handles ring topology communication for context parallelism."""

    def __init__(self, process_group: Optional[dist.ProcessGroup] = None):
        self.group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.next_rank = (self.rank + 1) % self.world_size
        self.prev_rank = (self.rank - 1) % self.world_size

    def send_recv_kv(
        self,
        k_send: torch.Tensor,
        v_send: torch.Tensor,
        k_recv: torch.Tensor,
        v_recv: torch.Tensor,
        direction: str = 'forward'
    ) -> List[dist.Work]:
        """
        Async send/recv of K,V tensors around the ring.

        Args:
            k_send: Key tensor to send
            v_send: Value tensor to send
            k_recv: Buffer for received keys
            v_recv: Buffer for received values
            direction: 'forward' (send to next, recv from prev) or
                      'backward' (send to prev, recv from next)

        Returns:
            List of async work handles to wait on
        """
        if direction == 'forward':
            send_rank = self.next_rank
            recv_rank = self.prev_rank
        else:
            send_rank = self.prev_rank
            recv_rank = self.next_rank

        ops = []
        # Send operations
        ops.append(dist.isend(k_send.contiguous(), send_rank, group=self.group))
        ops.append(dist.isend(v_send.contiguous(), send_rank, group=self.group))
        # Receive operations
        ops.append(dist.irecv(k_recv, recv_rank, group=self.group))
        ops.append(dist.irecv(v_recv, recv_rank, group=self.group))

        return ops

    @staticmethod
    def wait_all(ops: List[dist.Work]) -> None:
        """Wait for all async operations to complete."""
        for op in ops:
            op.wait()
