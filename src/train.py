import logging
import time
from typing import Any
from typing import Iterable, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.strategies import ParallelStrategy
from torch.utils.data import DataLoader
import torch

from src.algorithms.algorithm_base import AlgorithmBase
from src.datamodules.wair_d_base import WAIRDBaseDatamodule
from src.utils import EpochCounter, log_hyperparameters

log = logging.getLogger(__name__)


def train(config: DictConfig) -> None:
    epoch_counter = EpochCounter()
    gpus = config.trainer.devices
    multi_gpu = gpus == -1 or (isinstance(gpus, Iterable) and len(gpus) > 1) or (isinstance(gpus, int) and gpus > 1)
    
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: WAIRDBaseDatamodule = hydra.utils.instantiate(
        config.datamodule,
        epoch_counter=epoch_counter, multi_gpu=multi_gpu, drop_last=not config.algorithm.compiled.disable
    )
    # Quick dataset sizes
    try:
        train_size = len(getattr(datamodule, 'train_set')) if getattr(datamodule, 'train_set', None) is not None else 0
        val_size = 0
        if getattr(datamodule, 'val_set', None) is not None:
            val_obj = datamodule.val_set
            val_size = sum(len(v) for v in val_obj) if isinstance(val_obj, list) else len(val_obj)
        log.info(f"Dataset sizes: train={train_size}, val={val_size}")
    except Exception as e:
        log.warning(f"Unable to compute dataset sizes: {e}")
    
    log.info(f"Instantiating algorithm {config.algorithm._target_}")
    algorithm: AlgorithmBase = hydra.utils.instantiate(
        config.algorithm,
        epoch_counter=epoch_counter,
        network=None,  # instead, we give network_conf
        network_conf=(OmegaConf.to_yaml(config.network) if "network" in config else None),
        optimizer_conf=(OmegaConf.to_yaml(config.optimizer) if "optimizer" in config else None),
        scheduler_conf=(OmegaConf.to_yaml(config.scheduler) if "scheduler" in config else None)
    )
    
    # Init lightning callbacks
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    
    # Init lightning loggers
    loggers: list[Logger] = []
    if "loggers" in config:
        for name, lg_conf in config.loggers.items():
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger = hydra.utils.instantiate(lg_conf)
            loggers.append(logger)
    
    if "strategy" in config:
        log.info(f"Instantiating strategy <{config.strategy}>")
        strategy: Optional[ParallelStrategy] = hydra.utils.instantiate(config.strategy)
    else:
        if multi_gpu:
            log.error("In case of using multiple GPUs, you must provide a strategy")
        strategy = None
    
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, strategy=strategy or "auto", _convert_="partial"
    )
    
    log_hyperparameters(config=config, algorithm=algorithm, trainer=trainer)
    
    # Train the model
    # Improve matmul performance on Tensor Cores
    try:
        torch.set_float32_matmul_precision('medium')
        log.info("Set torch float32 matmul precision to 'medium'")
    except Exception:
        pass

    # Preflight: build a no-worker DataLoader and fetch one batch to surface slow IO early
    try:
        if getattr(datamodule, 'train_set', None) is not None and len(datamodule.train_set) > 0:
            preflight_bs = getattr(config.datamodule, 'batch_size', 1)
            t0 = time.perf_counter()
            pre_dl = DataLoader(datamodule.train_set, batch_size=preflight_bs, num_workers=0, shuffle=False)
            log.info(f"Preflight: created single-process DataLoader (batch_size={preflight_bs}) in {time.perf_counter() - t0:.2f}s")
            t1 = time.perf_counter()
            first_batch = next(iter(pre_dl))
            t_fetch = time.perf_counter() - t1
            shapes: list[Any] = []
            try:
                # Expect tuple (input, output, mask, meta)
                shapes = [tuple(x.shape) if hasattr(x, 'shape') else type(x) for x in first_batch[:3]]
            except Exception:
                pass
            log.info(f"Preflight: fetched first batch in {t_fetch:.2f}s; shapes={shapes}")
    except Exception as e:
        log.warning(f"Preflight failed: {e}")

    log.info("Starting training!")
    trainer.fit(algorithm, datamodule=datamodule, ckpt_path=config.ckpt_path)
    
    trainer.test(dataloaders=datamodule.test_dataloader())
