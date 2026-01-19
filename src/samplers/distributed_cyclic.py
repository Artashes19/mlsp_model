import math
from typing import Optional

import numpy as np
import torch.distributed as dist
from torch.utils.data import DistributedSampler


class DistributedCyclicSampler(DistributedSampler):
    """
    DDP-aware cyclic sampler that:
      - Provides a continuous, non-resetting stream of indices across epochs
      - Uses a single fixed random permutation (seeded) for determinism
      - Partitions work equally across ranks each epoch
      - Advances a shared base pointer by world_size * per_rank each epoch
    """
    
    def __init__(
        self,
        dataset,
        samples_per_epoch_total: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None
    ):
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        # Initialize as a DistributedSampler to avoid Lightning replacing our sampler
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False, drop_last=False)
        self._dataset_len = len(dataset)
        self._samples_per_epoch_total = max(0, int(samples_per_epoch_total))
        # Equalize steps across ranks for DDP; small overshoot vs total is acceptable
        self._per_rank = int(
            math.ceil(self._samples_per_epoch_total / float(self.num_replicas))
        ) if self.num_replicas > 0 else self._samples_per_epoch_total
        # Fixed shuffled order for cycling (uses global numpy seed from seed_everything)
        self._order = np.arange(self._dataset_len, dtype=np.int64)
        if self._dataset_len > 0:
            np.random.shuffle(self._order)
        # Global base pointer (for rank 0); ranks offset by +rank at iteration time
        self._base_pos = 0
    
    def __iter__(self):
        m = self._dataset_len
        if m == 0 or self._per_rank == 0:
            return iter(())
        start = self._base_pos
        # Yield this rank's strided slice
        for i in range(self._per_rank):
            gi = (start + i * self.num_replicas + self.rank) % m
            yield int(self._order[gi])
        # Advance global pointer by the total work done across all ranks
        self._base_pos = (self._base_pos + (self._per_rank * self.num_replicas) % m) % m
    
    def __len__(self):
        return self._per_rank
    
    def set_epoch(self, epoch: int):
        # No-op: keep continuous stream; do not reshuffle/reset
        return
    
    @property
    def position(self) -> int:
        return self._base_pos
    
    def set_position(self, pos: int):
        if self._dataset_len == 0:
            self._base_pos = 0
        else:
            self._base_pos = int(pos) % self._dataset_len
