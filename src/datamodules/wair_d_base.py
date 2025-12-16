import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler


class WAIRDBaseDatamodule(pl.LightningDataModule):
    
    collate_fn = None
    
    def __init__(self, batch_size: int, num_workers: int, drop_last: bool, multi_gpu: bool = False, *args, **kwargs):
        super().__init__()
        
        self._batch_size = batch_size
        # Coerce possibly-string env values (e.g., '8') to int
        try:
            self._num_workers = int(num_workers) if num_workers is not None else 0
        except Exception:
            self._num_workers = 0
        self._drop_last = drop_last
        
        self._train_set = None
        self._val_set = None
        self._test_set = None
        self.args = args
        self.kwargs = kwargs
        
        self._multi_gpu = multi_gpu
        if multi_gpu:
            self.prepare_data()
    
    def train_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self._train_set) if self._multi_gpu else None
        dl_kwargs = dict(
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            sampler=sampler,
            shuffle=None if self._multi_gpu else True,
            collate_fn=self.collate_fn,
            drop_last=self._drop_last,
            pin_memory=True,
        )
        if self._num_workers and self._num_workers > 0:
            dl_kwargs.update(persistent_workers=True, prefetch_factor=2)
        try:
            logging.getLogger(__name__).info(
                f"[dataloader/train] size={len(self._train_set) if self._train_set is not None else 0}, "
                f"batch_size={self._batch_size}, num_workers={self._num_workers}, "
                f"sampler={type(sampler).__name__ if sampler is not None else 'None'}, "
                f"multi_gpu={self._multi_gpu}"
            )
        except Exception:
            pass
        return DataLoader(self._train_set, **dl_kwargs)
    
    def val_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self._val_set, shuffle=False) if self._multi_gpu else None
        dl_kwargs = dict(
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            sampler=sampler,
            collate_fn=self.collate_fn,
            drop_last=self._drop_last,
            pin_memory=True,
        )
        if self._num_workers and self._num_workers > 0:
            dl_kwargs.update(persistent_workers=True, prefetch_factor=2)
        try:
            logging.getLogger(__name__).info(
                f"[dataloader/val] size={len(self._val_set) if self._val_set is not None else 0}, "
                f"batch_size={self._batch_size}, num_workers={self._num_workers}, "
                f"sampler={type(sampler).__name__ if sampler is not None else 'None'}, "
                f"multi_gpu={self._multi_gpu}"
            )
        except Exception:
            pass
        return DataLoader(self._val_set, **dl_kwargs)
    
    def test_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self._test_set, shuffle=False) if self._multi_gpu else None
        dl_kwargs = dict(
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            sampler=sampler,
            collate_fn=self.collate_fn,
            drop_last=self._drop_last,
            pin_memory=True,
        )
        if self._num_workers and self._num_workers > 0:
            dl_kwargs.update(persistent_workers=True, prefetch_factor=2)
        try:
            logging.getLogger(__name__).info(
                f"[dataloader/test] size={len(self._test_set) if self._test_set is not None else 0}, "
                f"batch_size={self._batch_size}, num_workers={self._num_workers}, "
                f"sampler={type(sampler).__name__ if sampler is not None else 'None'}, "
                f"multi_gpu={self._multi_gpu}"
            )
        except Exception:
            pass
        return DataLoader(self._test_set, **dl_kwargs)
    
    @property
    def train_set(self):
        return self._train_set
    
    @property
    def test_set(self):
        return self._test_set
    
    @property
    def val_set(self):
        return self._val_set
