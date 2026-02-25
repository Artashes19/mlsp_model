import os
from typing import Sequence

from pytorch_lightning.callbacks import ModelCheckpoint


class ScheduledSampleModelCheckpoint(ModelCheckpoint):
    """
    Save checkpoints at selected sample counts (converted to global steps).

    Retrieves batch_size from the datamodule at setup time and converts
    sample thresholds to step thresholds. Saves when trainer.global_step
    reaches a threshold.
    """
    
    def __init__(
        self,
        schedule_samples: Sequence[int],
        **kwargs,
    ) -> None:
        kwargs.setdefault("monitor", None)
        kwargs.setdefault("save_last", False)
        kwargs.setdefault("save_on_train_epoch_end", False)
        super().__init__(**kwargs)
        self._schedule_samples = tuple(schedule_samples)
        self._schedule_steps: set[int] = set()
    
    def setup(self, trainer, pl_module, stage=None) -> None:
        batch_size: int = trainer.datamodule._batch_size
        self._schedule_steps = {
            s // batch_size for s in self._schedule_samples
        }
    
    def _should_save_step(self, trainer) -> bool:
        step = int(getattr(trainer, "global_step", 0))
        return step in self._schedule_steps
    
    def on_validation_end(self, trainer, pl_module) -> None:
        return
    
    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs) -> None:
        return
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self._should_save_step(trainer):
            step = trainer.global_step
            filename = self.filename.format(step=step)
            filepath = os.path.join(self.dirpath, f"{filename}.ckpt")
            self._save_checkpoint(trainer, filepath)
            self._last_global_step_saved = step
