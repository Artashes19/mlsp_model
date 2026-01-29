from pytorch_lightning.callbacks import Callback


class ValMetricAliasLogger(Callback):
    def __init__(self, metric_map: dict[str, str]):
        self._metric_map = metric_map
        self._last_logged_epoch = None

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        logger = trainer.logger
        if logger is None:
            return
        epoch = trainer.current_epoch
        if self._last_logged_epoch == epoch:
            return
        metrics = trainer.callback_metrics
        alias_metrics = {}
        for src_key, dst_key in self._metric_map.items():
            if src_key in metrics:
                alias_metrics[dst_key] = metrics[src_key]
        if alias_metrics:
            logger.log_metrics(alias_metrics, epoch)
        self._last_logged_epoch = epoch
