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
        channels: str,
        train_augmentations: Optional[AugmentationPipeline],
        test_manifest_path: Optional[list[str]],
        val_sparse_ranges: Optional[list[list[float]]] = None,
        **kwargs
    ):
        # Validate and store channels configuration
        valid_channels = "rtdgfmps"
        if not channels:
            raise ValueError("channels cannot be empty")
        if len(channels) != len(set(channels)):
            raise ValueError(f"channels cannot contain duplicates: {channels}")
        invalid = set(channels) - set(valid_channels)
        if invalid:
            raise ValueError(f"Invalid channel letters: {invalid}. Valid letters: {valid_channels}")
        self.channels = channels
        
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
        
        # Store kwargs for dataset
        self.dataset_kwargs = kwargs
        self.dataset_kwargs["channels"] = channels
        
        # Initialize base LightningDataModule
        super().__init__()
        
        # Initialize dataloader settings
        self._batch_size = batch_size
        self._num_workers = int(num_workers) if num_workers is not None else 0
        self._multi_gpu = multi_gpu
        
        self._train_set = None
        self._test_sets: dict[str, PathlossDataset] = {}
        
        # Three ICASSP validation sets with different channel configs
        self._val_set_no_sparse = None       # sparse disabled
        self._val_set_no_trans_ref = None    # trans+ref disabled
        self._val_set_all_enabled = None     # all channels enabled
        self._val_sets: list[tuple[PathlossDataset, PathlossDataset, PathlossDataset]] = []
        
        # Three synthetic validation sets (when use_synthetic_train)
        self._synth_val_set_no_sparse = None
        self._synth_val_set_no_trans_ref = None
        self._synth_val_set_all_enabled = None
        
        self._data_prepared = False
        
        self.prepare_data()
    
    @staticmethod
    def get_inputs_list(
        freqs_mhz: list[int],
        freqs: list[int],
        manifest_path: str,
    ) -> list:
        """Load inputs from a manifest file. Manifest path is required."""
        t0 = time.perf_counter()
        log.info(
            f"[inputs] loading from manifest={manifest_path}, "
            f"freqs_mhz={list(freqs_mhz) if freqs_mhz else []}, freqs={list(freqs) if freqs else []}"
        )
        
        if not manifest_path:
            raise RuntimeError("manifest_path is required but was empty or None.")
        if not os.path.isfile(manifest_path):
            raise RuntimeError(f"manifest_path does not exist: {manifest_path}")
        
        inputs_list = []
        n_rows = 0
        n_synth = 0
        n_real = 0
        
        with open(manifest_path, "r", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                n_rows += 1
                # Common fields
                b = int(row.get("building", "0") or 0)
                ant = int(row.get("antenna", "0") or 0)
                # Resolve frequency index (1-based)
                f_idx_internal = None
                for key in ("freq_idx", "frequency_index"):
                    if row.get(key) not in (None, ""):
                        f_idx_internal = int(row.get(key))
                        break
                if f_idx_internal is None:
                    freq_mhz_val = float(row.get("freq_MHz", row.get("frequency_MHz")))
                    if freqs_mhz and len(freqs_mhz) > 0:
                        diffs = [abs(freq_mhz_val - float(m)) for m in freqs_mhz]
                        nearest = int(min(range(len(diffs)), key=lambda i: diffs[i]))
                        f_idx_internal = 1 + nearest
                    else:
                        f_idx_internal = 1
                sp = int(row.get("sample_index", row.get("sampling_position", 0)))
                # Filter by requested frequencies if provided
                if freqs and f_idx_internal not in freqs:
                    continue
                # Synthetic row
                npz_path = row.get("npz_file")
                json_path = row.get("json_file")
                if npz_path and json_path:
                    sample_name = row.get("file_name") or (
                        os.path.splitext(os.path.basename(npz_path))[0] if npz_path else None)
                    inputs_list.append(
                        {
                            "file_name": sample_name,
                            "npz_file": npz_path,
                            "json_file": json_path,
                            "ids": (b, ant, f_idx_internal, sp),
                        }
                    )
                    n_synth += 1
                    continue
                # ICASSP row
                input_file = row.get("input_file")
                output_file = row.get("output_file")
                position_file = row.get("position_file")
                radiation_pattern_file = row.get("radiation_pattern_file")
                sparse_file = row.get("sparse_file", "")
                if input_file and position_file and radiation_pattern_file:
                    sample_name = row.get("file_name") or os.path.basename(input_file)
                    freq_mhz = float(
                        row.get("freq_MHz", freqs_mhz[f_idx_internal - 1] if f_idx_internal else freqs_mhz[0])
                    )
                    entry = {
                        "file_name": sample_name,
                        "freq_MHz": freq_mhz,
                        "input_file": input_file,
                        "output_file": output_file or "",
                        "position_file": position_file,
                        "radiation_pattern_file": radiation_pattern_file,
                        "sampling_position": sp,
                        "ids": (b, ant, f_idx_internal, sp),
                    }
                    if sparse_file:
                        entry["sparse_file"] = sparse_file
                    inputs_list.append(entry)
                    n_real += 1
        
        dt = time.perf_counter() - t0
        log.info(
            f"[inputs] loaded from manifest={manifest_path} rows={n_rows}; "
            f"parsed: synthetic={n_synth}, real={n_real} (elapsed={dt:.2f}s)"
        )
        return inputs_list
    
    def _create_validation_sets(
        self,
        val_inputs: list,
        sparse_range: Optional[tuple[float, float]] = None,
    ) -> tuple[PathlossDataset, PathlossDataset, PathlossDataset]:
        """
        Create three validation datasets with different channel configurations.
        
        Args:
            val_inputs: List of input samples for validation.
        
        Returns:
            Tuple of (no_sparse, no_trans_ref, all_enabled) datasets:
            - no_sparse: sparse channel disabled
            - no_trans_ref: transmittance and reflectance disabled
            - all_enabled: all channels enabled
        """
        dataset_kwargs = dict(self.dataset_kwargs)
        if sparse_range is not None:
            dataset_kwargs["sparse_range"] = sparse_range
        
        # Val set 1: sparse disabled
        no_sparse = PathlossDataset(
            val_inputs,
            training=False,
            augmentations=None,
            inference=self.inference,
            force_drop_sparse=True,
            force_drop_trans_ref=False,
            **dataset_kwargs
        )
        # Val set 2: trans+ref disabled
        no_trans_ref = PathlossDataset(
            val_inputs,
            training=False,
            augmentations=None,
            inference=self.inference,
            force_drop_sparse=False,
            force_drop_trans_ref=True,
            **dataset_kwargs
        )
        # Val set 3: all enabled
        all_enabled = PathlossDataset(
            val_inputs,
            training=False,
            augmentations=None,
            inference=self.inference,
            force_drop_sparse=False,
            force_drop_trans_ref=False,
            **dataset_kwargs
        )
        return no_sparse, no_trans_ref, all_enabled
    
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
        icassp_val_inputs = self.get_inputs_list(
            freqs_mhz=self.freqs_mhz,
            freqs=self.freqs,
            manifest_path=self.val_manifest_path,
        )
        
        # Create ICASSP validation datasets (always)
        if self._val_sparse_ranges:
            self._val_sets = []
            for sparse_range in self._val_sparse_ranges:
                self._val_sets.append(
                    self._create_validation_sets(
                        val_inputs=icassp_val_inputs,
                        sparse_range=sparse_range,
                    )
                )
            (
                self._val_set_no_sparse,
                self._val_set_no_trans_ref,
                self._val_set_all_enabled,
            ) = self._val_sets[0]
        else:
            (
                self._val_set_no_sparse,
                self._val_set_no_trans_ref,
                self._val_set_all_enabled,
            ) = self._create_validation_sets(val_inputs=icassp_val_inputs)
        
        if self.inference:
            # For inference, load from test_manifest_path list
            if self.test_manifest_path:
                for test_path in self.test_manifest_path:
                    if test_path and os.path.isfile(test_path):
                        test_inputs = self.get_inputs_list(
                            freqs_mhz=self.freqs_mhz,
                            freqs=self.freqs,
                            manifest_path=test_path,
                        )
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
            log.info(f"Prepared inference datasets: val_sets={len(icassp_val_inputs)}, test_sets={len(self._test_sets)}")
        else:
            if self.use_synthetic_train:
                # Synthetic training: load from synthetic_manifest_path
                train_inputs = self.get_inputs_list(
                    freqs_mhz=self.freqs_mhz,
                    freqs=self.freqs,
                    manifest_path=self.synthetic_manifest_path,
                )
                # Apply synthetic_limit if set
                if self.synthetic_limit is not None and self.synthetic_limit > 0:
                    train_inputs = sorted(train_inputs, key=_sort_key)[:self.synthetic_limit]
                
                # Also load synthetic validation
                synth_val_inputs = self.get_inputs_list(
                    freqs_mhz=self.freqs_mhz,
                    freqs=self.freqs,
                    manifest_path=self.synthetic_val_manifest_path,
                )
            else:
                # Real training: select manifest based on use_small_train flag
                if self.use_small_train:
                    real_train_manifest = self.train_small_manifest_path
                else:
                    real_train_manifest = self.train_manifest_path
                train_inputs = self.get_inputs_list(
                    freqs_mhz=self.freqs_mhz,
                    freqs=self.freqs,
                    manifest_path=real_train_manifest,
                )
                synth_val_inputs = None
            
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
            
            # Create 3 synthetic validation datasets (only for synthetic training)
            if synth_val_inputs:
                (
                    self._synth_val_set_no_sparse,
                    self._synth_val_set_no_trans_ref,
                    self._synth_val_set_all_enabled,
                ) = self._create_validation_sets(val_inputs=synth_val_inputs)
                log.info(f"[datasets] prepared synthetic validation sets: n={len(self._synth_val_set_no_sparse)}")
            
            # Create test datasets from test_manifest_path list
            if self.test_manifest_path:
                for test_path in self.test_manifest_path:
                    if test_path and os.path.isfile(test_path):
                        test_inputs = self.get_inputs_list(
                            freqs_mhz=self.freqs_mhz,
                            freqs=self.freqs,
                            manifest_path=test_path,
                        )
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
            f"val_sets={len(self._val_set_no_sparse) if self._val_set_no_sparse is not None else 0}, "
            f"synth_val_sets={len(self._synth_val_set_no_sparse) if self._synth_val_set_no_sparse is not None else 0}, "
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
    def val_set_no_sparse(self):
        return self._val_set_no_sparse
    
    @property
    def val_set_no_trans_ref(self):
        return self._val_set_no_trans_ref
    
    @property
    def val_set_all_enabled(self):
        return self._val_set_all_enabled
    
    @property
    def synth_val_set_no_sparse(self):
        return self._synth_val_set_no_sparse
    
    @property
    def synth_val_set_no_trans_ref(self):
        return self._synth_val_set_no_trans_ref
    
    @property
    def synth_val_set_all_enabled(self):
        return self._synth_val_set_all_enabled
    
    def val_dataloader(self) -> list[DataLoader]:
        """
        Returns validation dataloaders with different channel configurations.
        
        Order:
        - Index 0: ICASSP, sparse disabled
        - Index 1: ICASSP, trans+ref disabled
        - Index 2: ICASSP, all enabled
        - Index 3: Synthetic, sparse disabled (only when use_synthetic_train)
        - Index 4: Synthetic, trans+ref disabled (only when use_synthetic_train)
        - Index 5: Synthetic, all enabled (only when use_synthetic_train)
        """
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
        
        # ICASSP validation sets (always)
        if self._val_sets:
            for no_sparse, no_trans_ref, all_enabled in self._val_sets:
                dataloaders.append(_make_dataloader(no_sparse))
                dataloaders.append(_make_dataloader(no_trans_ref))
                dataloaders.append(_make_dataloader(all_enabled))
        else:
            if self._val_set_no_sparse:
                dataloaders.append(_make_dataloader(self._val_set_no_sparse))
            if self._val_set_no_trans_ref:
                dataloaders.append(_make_dataloader(self._val_set_no_trans_ref))
            if self._val_set_all_enabled:
                dataloaders.append(_make_dataloader(self._val_set_all_enabled))
        
        # Synthetic validation sets (only when use_synthetic_train, 3 dataloaders)
        if self._synth_val_set_no_sparse:
            dataloaders.append(_make_dataloader(self._synth_val_set_no_sparse))
        if self._synth_val_set_no_trans_ref:
            dataloaders.append(_make_dataloader(self._synth_val_set_no_trans_ref))
        if self._synth_val_set_all_enabled:
            dataloaders.append(_make_dataloader(self._synth_val_set_all_enabled))
        
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
