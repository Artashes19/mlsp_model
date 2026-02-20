import csv
import logging
import os
import re
import time
from typing import Optional

import pytorch_lightning as pl
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from src.datamodules.datasets import PathlossDataset
from src.samplers import DistributedCyclicSampler
from src.utils.indoor.augmentations import AugmentationPipeline

log = logging.getLogger(__name__)


class IndoorDatamodule(pl.LightningDataModule):
    
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        data_dir: str,
        synthetic_dir: str,
        freqs_mhz: list[int],
        freqs: list[int],
        inference: bool,
        use_synthetic_train: bool,
        use_small_train: bool,
        train_manifest_path: str,
        train_small_manifest_path: str,
        val_manifest_path: str,
        synthetic_manifest_path: str,
        synthetic_val_manifest_path: str,
        synthetic_limit: Optional[int],
        train_samples_per_epoch: int,
        multi_gpu: bool,
        train_augmentations: Optional[AugmentationPipeline],
        test_manifest_path: Optional[list[str]],
        val_modalities: list[str] = None,
        val_sparse_ranges: Optional[list[list[float]]] = None,
        **kwargs
    ):
        self.freqs_mhz = freqs_mhz
        self.freqs = freqs
        self.data_dir = data_dir
        self.synthetic_dir = synthetic_dir
        self.inference = inference
        
        # Training augmentations config (Hydra instantiable dict or None)
        self.train_augmentations = train_augmentations
        
        # Experiment controls
        self.use_synthetic_train = use_synthetic_train
        self.use_small_train = use_small_train
        self.train_manifest_path = train_manifest_path
        self.train_small_manifest_path = train_small_manifest_path
        self.val_manifest_path = val_manifest_path
        self.synthetic_manifest_path = synthetic_manifest_path
        self.synthetic_val_manifest_path = synthetic_val_manifest_path
        self.test_manifest_path = test_manifest_path
        self.synthetic_limit = int(synthetic_limit) if synthetic_limit is not None else None
        self.train_samples_per_epoch = int(train_samples_per_epoch) if train_samples_per_epoch is not None else None
        
        # Validation modality types: "RT" (no sparse), "S" (sparse only), "RTS" (all)
        if val_modalities is None:
            raise ValueError("val_modalities must be provided (e.g. ['RT'], ['RTS'], ['RT', 'S', 'RTS'])")
        self._val_modalities = list(val_modalities)
        
        self._val_sparse_ranges = None
        if val_sparse_ranges:
            ranges = []
            for entry in val_sparse_ranges:
                if entry is None:
                    continue
                if len(entry) != 2:
                    raise ValueError("val_sparse_ranges entries must be length 2")
                lo = float(entry[0])
                hi = float(entry[1])
                ranges.append((lo, hi))
            if ranges:
                self._val_sparse_ranges = ranges
        
        # Validate: S and RTS modalities require sparse ranges
        needs_sparse = any(m in ("S", "RTS") for m in self._val_modalities)
        if needs_sparse and not self._val_sparse_ranges:
            raise ValueError("val_sparse_ranges required when val_modalities contains 'S' or 'RTS'")
        
        # Store kwargs for dataset
        self.dataset_kwargs = kwargs
        
        # Initialize base LightningDataModule
        super().__init__()
        
        # Initialize dataloader settings
        self._batch_size = batch_size
        self._num_workers = int(num_workers) if num_workers is not None else 0
        self._multi_gpu = multi_gpu
        
        self._train_set = None
        self._test_sets: dict[str, PathlossDataset] = {}
        
        # Ordered list of all validation datasets
        self._val_datasets: list[PathlossDataset] = []
        
        self._data_prepared = False
        
        self.prepare_data()
    
    @staticmethod
    def get_inputs_list(manifest_path: str) -> list:
        """Load inputs from a manifest file."""
        t0 = time.perf_counter()
        log.info(f"[inputs] loading from manifest={manifest_path}")
        
        if not manifest_path:
            raise RuntimeError("manifest_path is required but was empty or None.")
        if not os.path.isfile(manifest_path):
            raise RuntimeError(f"manifest_path does not exist: {manifest_path}")
        
        inputs_list = []
        n_synth = 0
        n_real = 0
        
        with open(manifest_path, "r", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                freq_mhz = float(row["freq_mhz"])
                
                # Synthetic row (has npz_file and json_file)
                npz_path = row.get("npz_file")
                json_path = row.get("json_file")
                if npz_path and json_path:
                    inputs_list.append(
                        {
                            "file_name": row.get("file_name") or os.path.splitext(os.path.basename(npz_path))[0],
                            "freq_mhz": freq_mhz,
                            "npz_file": npz_path,
                            "json_file": json_path,
                        }
                    )
                    n_synth += 1
                    continue
                # ICASSP row (has input_file, position_file, radiation_pattern_file)
                input_file = row.get("input_file")
                position_file = row.get("position_file")
                radiation_pattern_file = row.get("radiation_pattern_file")
                if input_file and position_file and radiation_pattern_file:
                    entry = {
                        "file_name": row.get("file_name") or os.path.basename(input_file),
                        "freq_mhz": freq_mhz,
                        "input_file": input_file,
                        "output_file": row.get("output_file") or "",
                        "position_file": position_file,
                        "radiation_pattern_file": radiation_pattern_file,
                        "sample_index": int(row["sample_index"]),
                    }
                    sparse_file = row.get("sparse_file", "")
                    if sparse_file:
                        entry["sparse_file"] = sparse_file
                    inputs_list.append(entry)
                    n_real += 1
        
        dt = time.perf_counter() - t0
        log.info(f"[inputs] loaded {len(inputs_list)} entries (synthetic={n_synth}, real={n_real}) in {dt:.2f}s")
        return inputs_list
    
    def _create_val_dataset(
        self,
        val_inputs: list,
        force_drop_sparse: bool,
        force_drop_trans_ref: bool,
        sparse_range: Optional[tuple[float, float]] = None,
    ) -> PathlossDataset:
        """Create a single validation dataset with specific modality configuration."""
        dataset_kwargs = dict(self.dataset_kwargs)
        if sparse_range is not None:
            dataset_kwargs["sparse_range"] = sparse_range
        
        return PathlossDataset(
            val_inputs,
            training=False,
            augmentations=None,
            inference=self.inference,
            force_drop_sparse=force_drop_sparse,
            force_drop_trans_ref=force_drop_trans_ref,
            **dataset_kwargs
        )
    
    def _create_val_datasets_for_modalities(
        self,
        val_inputs: list,
    ) -> list[PathlossDataset]:
        """
        Create validation datasets based on val_modalities config.

        Order: RT first, then S per sparse range, then RTS per sparse range.
        """
        datasets = []
        for modality in self._val_modalities:
            if modality == "RT":
                # No sparse, has reflectance+transmittance
                datasets.append(
                    self._create_val_dataset(
                        val_inputs,
                        force_drop_sparse=True,
                        force_drop_trans_ref=False,
                    )
                )
            elif modality == "S":
                # Sparse only (no trans+ref), one per sparse range
                for sparse_range in self._val_sparse_ranges:
                    datasets.append(
                        self._create_val_dataset(
                            val_inputs,
                            force_drop_sparse=False,
                            force_drop_trans_ref=True,
                            sparse_range=sparse_range,
                        )
                    )
            elif modality == "RTS":
                # All modalities enabled, one per sparse range
                for sparse_range in self._val_sparse_ranges:
                    datasets.append(
                        self._create_val_dataset(
                            val_inputs,
                            force_drop_sparse=False,
                            force_drop_trans_ref=False,
                            sparse_range=sparse_range,
                        )
                    )
            else:
                raise ValueError(f"Unknown val_modality: {modality!r}. Must be 'RT', 'S', or 'RTS'.")
        return datasets
    
    def prepare_data(self) -> None:
        if self._data_prepared:
            return
        
        t0_prepare = time.perf_counter()
        
        def _ids_of(obj):
            return getattr(obj, "ids", obj.get("ids") if isinstance(obj, dict) else None)
        
        def _sort_key(obj):
            ids = _ids_of(obj)
            if ids is None:
                return (str(getattr(obj, "file_name", "")),)
            return tuple(ids)
        
        # Always load ICASSP validation (regardless of inference flag)
        icassp_val_inputs = self.get_inputs_list(manifest_path=self.val_manifest_path)
        
        # Create ICASSP validation datasets based on val_modalities
        self._val_datasets = self._create_val_datasets_for_modalities(icassp_val_inputs)
        n_icassp_val = len(self._val_datasets)
        
        if self.inference:
            # For inference, load from test_manifest_path list
            if self.test_manifest_path:
                for test_path in self.test_manifest_path:
                    if test_path and os.path.isfile(test_path):
                        test_inputs = self.get_inputs_list(manifest_path=test_path)
                        test_dataset = PathlossDataset(
                            test_inputs,
                            training=False,
                            augmentations=None,
                            inference=True,
                            force_drop_sparse=False,
                            force_drop_trans_ref=False,
                            **self.dataset_kwargs
                        )
                        # Extract descriptive name from manifest filename
                        # mlsp_test_rate0.02.csv -> test_0.02
                        # icassp_test_Task_1.csv -> test
                        basename = os.path.basename(test_path)
                        rate_match = re.search(r"rate(\d+\.?\d*)", basename)
                        if rate_match:
                            test_name = f"test_{rate_match.group(1)}"
                        else:
                            test_name = "test"
                        self._test_sets[test_name] = test_dataset
                        log.info(f"[datasets] created test set '{test_name}' from {test_path}: n={len(test_dataset)}")
            log.info(f"Prepared inference datasets: val_sets={n_icassp_val}, test_sets={len(self._test_sets)}")
        else:
            synth_val_inputs = None
            if self.use_synthetic_train:
                # Synthetic training: load from synthetic_manifest_path
                train_inputs = self.get_inputs_list(manifest_path=self.synthetic_manifest_path)
                # Apply synthetic_limit if set
                if self.synthetic_limit is not None and self.synthetic_limit > 0:
                    train_inputs = sorted(train_inputs, key=_sort_key)[:self.synthetic_limit]
                
                # Also load synthetic validation
                synth_val_inputs = self.get_inputs_list(manifest_path=self.synthetic_val_manifest_path)
            else:
                # Real training: select manifest based on use_small_train flag
                if self.use_small_train:
                    real_train_manifest = self.train_small_manifest_path
                else:
                    real_train_manifest = self.train_manifest_path
                train_inputs = self.get_inputs_list(manifest_path=real_train_manifest)
            
            log.info(
                f"[datasets] train={len(train_inputs)}, icassp_val={len(icassp_val_inputs)}, "
                f"synth_val={len(synth_val_inputs) if synth_val_inputs else 0}, "
                f"use_synthetic_train={self.use_synthetic_train}"
            )
            
            # Create training dataset (use random dropout via None)
            self._train_set = PathlossDataset(
                train_inputs,
                training=True,
                augmentations=self.train_augmentations,
                inference=False,
                force_drop_sparse=None,
                force_drop_trans_ref=None,
                **self.dataset_kwargs
            )
            
            # Create synthetic validation datasets (same modality configs as ICASSP)
            if synth_val_inputs:
                synth_val_datasets = self._create_val_datasets_for_modalities(synth_val_inputs)
                self._val_datasets.extend(synth_val_datasets)
                log.info(f"[datasets] prepared synthetic validation sets: n={len(synth_val_datasets)}")
            
            # Create test datasets from test_manifest_path list
            if self.test_manifest_path:
                for test_path in self.test_manifest_path:
                    if test_path and os.path.isfile(test_path):
                        test_inputs = self.get_inputs_list(manifest_path=test_path)
                        test_dataset = PathlossDataset(
                            test_inputs,
                            training=False,
                            augmentations=None,
                            inference=True,
                            force_drop_sparse=False,
                            force_drop_trans_ref=False,
                            **self.dataset_kwargs
                        )
                        # Extract descriptive name from manifest filename
                        # mlsp_test_rate0.02.csv -> test_0.02
                        # icassp_test_Task_1.csv -> test
                        basename = os.path.basename(test_path)
                        rate_match = re.search(r"rate(\d+\.?\d*)", basename)
                        if rate_match:
                            test_name = f"test_{rate_match.group(1)}"
                        else:
                            test_name = "test"
                        self._test_sets[test_name] = test_dataset
                        log.info(f"[datasets] created test set '{test_name}' from {test_path}: n={len(test_dataset)}")
        
        log.info(
            f"[prepare_data] done in {(time.perf_counter() - t0_prepare):.2f}s "
            f"(train_set={len(self._train_set) if self._train_set is not None else 0}, "
            f"val_datasets={len(self._val_datasets)}, "
            f"test_sets={len(self._test_sets)})"
        )
        
        self._data_prepared = True
    
    @property
    def train_set(self):
        return self._train_set
    
    @property
    def test_sets(self) -> dict[str, PathlossDataset]:
        return self._test_sets
    
    @property
    def val_datasets(self) -> list[PathlossDataset]:
        return self._val_datasets
    
    def val_dataloader(self) -> list[DataLoader]:
        """Returns validation dataloaders, one per entry in val_datasets."""
        dataloaders = []
        
        def _make_dataloader(dataset):
            sampler = DistributedSampler(dataset, shuffle=False) if self._multi_gpu else None
            dl_kwargs = dict(
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                sampler=sampler,
                drop_last=False,
                pin_memory=True,
            )
            if self._num_workers and self._num_workers > 0:
                dl_kwargs.update(persistent_workers=True, prefetch_factor=2)
            return DataLoader(dataset, **dl_kwargs)
        
        for dataset in self._val_datasets:
            dataloaders.append(_make_dataloader(dataset))
        
        return dataloaders
    
    def test_dataloader(self) -> list[DataLoader]:
        """Returns test dataloaders, one per test manifest."""
        dataloaders = []
        for test_set in self._test_sets.values():
            sampler = DistributedSampler(test_set, shuffle=False) if self._multi_gpu else None
            dl_kwargs = dict(
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                sampler=sampler,
                drop_last=False,
                pin_memory=True,
            )
            if self._num_workers and self._num_workers > 0:
                dl_kwargs.update(persistent_workers=True, prefetch_factor=2)
            dataloaders.append(DataLoader(test_set, **dl_kwargs))
        return dataloaders
    
    def train_dataloader(self) -> DataLoader:
        # Determine rank/world size if DDP is active
        def _dist_info():
            if dist.is_available() and dist.is_initialized():
                try:
                    return dist.get_rank(), dist.get_world_size()
                except Exception:
                    return 0, 1
            return 0, 1
        
        rank, world_size = _dist_info()
        
        sampler = DistributedCyclicSampler(
            self._train_set,
            samples_per_epoch_total=int(self.train_samples_per_epoch),
            num_replicas=world_size,
            rank=rank,
        )
        dl_kwargs = dict(
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )
        if self._num_workers and self._num_workers > 0:
            dl_kwargs.update(persistent_workers=True, prefetch_factor=8)
        try:
            log.info(
                f"[dataloader/train] size={len(self._train_set) if self._train_set is not None else 0}, "
                f"batch_size={self._batch_size}, num_workers={self._num_workers}, "
                f"sampler={type(sampler).__name__}, multi_gpu={self._multi_gpu}, "
                f"samples_per_epoch_total={self.train_samples_per_epoch}"
            )
        except Exception:
            pass
        return DataLoader(self._train_set, **dl_kwargs)
