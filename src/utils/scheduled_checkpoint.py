import os
from typing import Iterable, Sequence

from pytorch_lightning.callbacks import ModelCheckpoint


class ScheduledEpochModelCheckpoint(ModelCheckpoint):
    """
    Save checkpoints only at selected epochs plus the final epoch.

    Epoch indices are 0-based (i.e., epoch 0 -> first epoch).
    """

    def __init__(self, schedule_epochs: Sequence[int] | None = None, **kwargs) -> None:
        # Disable base-class monitoring logic; we control saving explicitly.
        kwargs.setdefault("monitor", None)
        kwargs.setdefault("save_last", False)
        super().__init__(**kwargs)
        self._schedule = {int(e) for e in (schedule_epochs or [])}

    def _should_save_epoch(self, trainer) -> bool:
        epoch = int(getattr(trainer, "current_epoch", 0))
        max_epochs = getattr(trainer, "max_epochs", None)
        if epoch in self._schedule:
            return True
        if max_epochs is not None and epoch == max_epochs - 1:
            return True
        return False

    def on_validation_end(self, trainer, pl_module) -> None:
        # Disable validation-based saving in the base class;
        # we only save on selected train epochs.
        return

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs) -> None:
        if self._should_save_epoch(trainer):
            return super().on_train_epoch_end(trainer, pl_module, *args, **kwargs)


