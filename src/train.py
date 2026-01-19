import json
import logging
import os
import time
from typing import Iterable, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.strategies import ParallelStrategy

from src.algorithms.algorithm_base import AlgorithmBase
from src.algorithms.mlsp import MLSP
from src.datamodules.mlsp import MLSPDatamodule
from src.utils import EpochCounter, load_experiment_config, log_hyperparameters

log = logging.getLogger(__name__)


def clone_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def train_prep(config: DictConfig, project_root: str):
    # Orchestrated multi-experiment run
    raw_exps = config.get("exps", "")
    exp_list = [str(x).strip() for x in raw_exps if str(x).strip()]
    
    for exp in exp_list:
        cfg_e = clone_cfg(config["exps"][exp])
        config["exps"][exp] = load_experiment_config(cfg_e, config_root=os.path.join(project_root, "configs/exps"))
    
    # Fast-dev toggle
    fast_dev = bool(config.get("fast_dev")) or bool(os.environ.get("FAST_DEV"))
    if fast_dev:
        log.info("[orchestrator] fast_dev enabled: will cap epochs/batches for quick smoke run")
    for exp in exp_list:
        cfg_e = clone_cfg(config["exps"][exp])
        log.info(
            f"[trainer@{exp}] devices={cfg_e.trainer.devices}, accelerator={cfg_e.trainer.accelerator}, "
            f"precision={cfg_e.trainer.precision}, max_epochs={cfg_e.trainer.max_epochs}"
        )
        
        if fast_dev:
            cfg_e.trainer.max_epochs = 1
            cfg_e.trainer.limit_train_batches = 4
            cfg_e.trainer.limit_val_batches = 2
        
        # Enforce required data roots (no fallbacks)
        data_dir_req = cfg_e.datamodule.get("data_dir")
        data_dir_req = os.path.expanduser(str(data_dir_req)) if data_dir_req is not None else ""
        if not data_dir_req or not os.path.isdir(data_dir_req):
            raise RuntimeError(
                f"datamodule.data_dir must point to an existing ICASSP root. Got: {cfg_e.datamodule.get('data_dir')}"
            )
        if exp == "e2":
            synth_dir_req = cfg_e.datamodule.get("synthetic_dir")
            synth_dir_req = os.path.expanduser(str(synth_dir_req)) if synth_dir_req is not None else ""
            if not synth_dir_req or not os.path.isdir(synth_dir_req):
                raise RuntimeError(
                    f"datamodule.synthetic_dir must point to an existing synthetic root for e2. Got: {cfg_e.datamodule.get('synthetic_dir')}"
                )
        
        # Summarize datamodule plan for this experiment
        try:
            dm = cfg_e.datamodule
            plan = dict(
                use_synthetic_train=bool(dm.get("use_synthetic_train", False)),
                train_manifest_path=dm.get("train_manifest_path", None),
                val_manifest_path=dm.get("val_manifest_path", None),
                synthetic_manifest_path=dm.get("synthetic_manifest_path", None),
                data_dir=dm.get("data_dir", None),
                synthetic_dir=dm.get("synthetic_dir", None),
            )
            log.info(f"[datamodule@{exp}] plan={plan}")
        except Exception:
            pass
        
        t0 = time.perf_counter()
        best = train(cfg_e)
        log.info(f"[train@{exp}] finished in {(time.perf_counter() - t0):.2f}s; best_checkpoint={best}")
    return None


def train(config: DictConfig) -> str | None:
    epoch_counter = EpochCounter()
    start_time = time.time()
    gpus = config.trainer.devices
    multi_gpu = gpus == -1 or (isinstance(gpus, Iterable) and len(gpus) > 1) or (isinstance(gpus, int) and gpus > 1)
    
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: MLSPDatamodule = hydra.utils.instantiate(
        config.datamodule,
        epoch_counter=epoch_counter,
        multi_gpu=multi_gpu,
    )
    # DEBUG: Print actual batch_size being used
    log.info(f"[DEBUG] ACTUAL batch_size={datamodule._batch_size}, num_workers={datamodule._num_workers}")
    # Summarize datasets if available
    try:
        tr_n = len(datamodule.train_set) if datamodule.train_set is not None else 0
        va_n = len(datamodule.val_set) if datamodule.val_set is not None else 0
        te_n = len(datamodule.test_set) if datamodule.test_set is not None else 0
        log.info(f"[datamodule] prepared datasets: train={tr_n}, val={va_n}, test={te_n}")
    except Exception:
        pass
    
    log.info(f"Instantiating algorithm {config.algorithm._target_}")
    ft_conf = config.algorithm.get("finetune")
    if ft_conf and bool(ft_conf["enable"]):
        ckpt_ft = os.path.abspath(str(config["ckpt_path"]))
        if not ckpt_ft:
            raise RuntimeError("Finetune is enabled but no ckpt_path was provided.")
        if not os.path.isfile(ckpt_ft):
            raise RuntimeError(f"Finetune checkpoint not found: {ckpt_ft}")
        # Recreate MLSP via Lightning's load_from_checkpoint with current config
        compiled_obj = hydra.utils.instantiate(config.algorithm.compiled)
        algorithm: AlgorithmBase = MLSP.load_from_checkpoint(
            ckpt_ft,
            strict=False,
            out_norm=float(config.algorithm["out_norm"]),
            use_sip2net=bool(config.algorithm["use_sip2net"]),
            sip2net_params=(OmegaConf.to_container(
                config.algorithm["sip2net_params"], resolve=True
            ) if "sip2net_params" in config.algorithm else {}),
            compiled=compiled_obj,
            optimizer_conf=(OmegaConf.to_yaml(config.optimizer) if "optimizer" in config else None),
            scheduler_conf=(OmegaConf.to_yaml(config.scheduler) if "scheduler" in config else None),
            network=None,
            network_conf=(OmegaConf.to_yaml(config.network) if "network" in config else None),
            gpu=None,
            finetune=OmegaConf.to_container(ft_conf, resolve=True),
            epoch_counter=epoch_counter,
        )
        # Optional: capture reference weights for L2-SP
        if bool(ft_conf["l2sp"]["enable"]):
            algorithm._capture_pretrained_reference()
        log.info(f"[algorithm] loaded from checkpoint (finetune): {ckpt_ft}")
    else:
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
        cb_names = []
        for name, cb_conf in config.callbacks.items():
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
            cb_names.append(name)
        log.info(f"[trainer] total callbacks={len(callbacks)} names={cb_names}")
    
    # Init lightning loggers
    loggers: list[Logger] = []
    if "loggers" in config:
        lg_names = []
        for name, lg_conf in config.loggers.items():
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger = hydra.utils.instantiate(lg_conf)
            loggers.append(logger)
            lg_names.append(name)
        log.info(f"[trainer] total loggers={len(loggers)} names={lg_names}")
    
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
    try:
        log.info(
            f"[trainer] devices={config.trainer.devices}, accelerator={config.trainer.accelerator}, "
            f"precision={config.trainer.precision}, max_epochs={config.trainer.max_epochs}, "
            f"default_root_dir={getattr(trainer, 'default_root_dir', None)}"
        )
    except Exception:
        pass
    
    if config["ckpt_path"] is not None and config["load_weights_only"]:
        log.info(f"Loading weights from {config['ckpt_path']}")
        ckpt = torch.load(config["ckpt_path"])
        algorithm.load_state_dict(ckpt['state_dict'])
        config["ckpt_path"] = None
    
    log_hyperparameters(config=config, algorithm=algorithm, trainer=trainer)
    
    # Train the model
    log.info("Starting training!")
    fit_t0 = time.time()
    trainer.fit(algorithm, datamodule=datamodule, ckpt_path=config.ckpt_path)
    log.info(f"Finished training in {time.time() - fit_t0:.2f}s")
    
    # Retrieve best checkpoint path if available
    best_path = None
    try:
        for cb in trainer.checkpoint_callbacks:
            p = getattr(cb, 'best_model_path', None)
            if p:
                best_path = p
                break
    except Exception:
        best_path = None
    
    # Persist simple results.json into experiment dir if default_root_dir hints at it
    try:
        duration_sec = time.time() - start_time
        # Collect final callback metrics as floats
        metrics = {}
        for k, v in (trainer.callback_metrics or {}).items():
            try:
                metrics[k] = float(v.detach().cpu()) if hasattr(v, 'detach') else float(v)
            except Exception:
                pass
        # Determine destination: parent of default_root_dir if endswith '/pl', else default_root_dir
        droot = getattr(trainer, 'default_root_dir', None) or None
        if trainer.is_global_zero and isinstance(droot, str) and droot:
            parent = os.path.dirname(droot.rstrip('/'))
            out_dir = parent if os.path.basename(droot.rstrip('/')) == 'pl' else droot
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'results.json'), 'w', encoding='utf-8') as fp:
                # Dataset sizes for traceability
                try:
                    tr_n = len(datamodule.train_set) if datamodule.train_set is not None else 0
                    va_n = len(datamodule.val_set) if datamodule.val_set is not None else 0
                    te_n = len(datamodule.test_set) if datamodule.test_set is not None else 0
                except Exception:
                    tr_n = va_n = te_n = 0
                json.dump(
                    {
                        'best_checkpoint': best_path,
                        'duration_sec': duration_sec,
                        'metrics': metrics,
                        'dataset': {
                            'train_size': tr_n,
                            'val_size': va_n,
                            'test_size': te_n,
                        }
                    }, fp, indent=2
                )
    except Exception:
        pass
    return best_path
