import os

from pytorch_lightning.callbacks import ModelCheckpoint


class EveryNEpochModelCheckpoint(ModelCheckpoint):
    """
    Save checkpoints every N epochs plus the final epoch.

    Epoch indices are 0-based (i.e., epoch 0 -> first epoch).
    """

    def __init__(self, every_n_epochs: int, **kwargs) -> None:
        kwargs.setdefault("monitor", None)
        kwargs.setdefault("save_last", False)
        kwargs.setdefault("save_on_train_epoch_end", True)
        super().__init__(**kwargs)
        if every_n_epochs <= 0:
            raise ValueError("every_n_epochs must be > 0")
        self._every_n_epochs = int(every_n_epochs)

    def _should_save_epoch(self, trainer) -> bool:
        epoch = int(getattr(trainer, "current_epoch", 0))
        max_epochs = getattr(trainer, "max_epochs", None)
        if (epoch + 1) % self._every_n_epochs == 0:
            return True
        if max_epochs is not None and epoch == max_epochs - 1:
            return True
        return False

    def on_validation_end(self, trainer, pl_module) -> None:
        return

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs) -> None:
        if self._should_save_epoch(trainer):
            epoch = trainer.current_epoch
            filename = self.filename.format(epoch=epoch)
            filepath = os.path.join(self.dirpath, f"{filename}.ckpt")
            self._save_checkpoint(trainer, filepath)
            self._last_global_step_saved = trainer.global_step
