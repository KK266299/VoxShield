# file: src/datasets/base_builder.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Tuple
from abc import ABC, abstractmethod
import warnings, random
from ..utils.logger import get_logger
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, ConcatDataset
from omegaconf import DictConfig, OmegaConf
from ..utils.config import get_config, require_config
from ..registry import get_dataset_builder
from .uekey_dataset import UEConcatDataset

class BaseDatasetBuilder(ABC):
    _ALLOWED = {"train", "val", "test"}
    _ALIASES = {"validate":"val","validation":"val","dev":"val","train":"train","test":"test"}
    _LOADER_ARG_KEYS = {
        "batch_size", "num_workers", "pin_memory", "drop_last",
        "prefetch_factor", "persistent_workers", "sampler", "shuffle",
        "timeout", "generator", "worker_init_fn", "collate_fn"
    }

    def __init__(self, config: DictConfig):
        self.config = config
        self._datasets: Dict[str, Dataset] = {}
        self._loaders: Dict[str, DataLoader] = {}
        self.logger = get_logger()
        
        # ---- common DataLoader configuration (moved to Base) ----
        tcfg = get_config(config, "training", OmegaConf.create({}))
        self.batch_size: int = int(get_config(tcfg, "batch_size", 32))
        self.eval_batch_size: int = int(get_config(tcfg, "eval_batch_size", self.batch_size))
        self.num_workers: int = int(get_config(tcfg, "num_workers", 4))
        self.pin_memory: bool = bool(get_config(tcfg, "pin_memory", True))
        self.persistent_workers: bool = bool(get_config(tcfg, "persistent_workers", self.num_workers > 0))
        self.prefetch_factor: Optional[int] = get_config(tcfg, "prefetch_factor", None)
        self.timeout: float = float(get_config(tcfg, "timeout", 0))
        self._deterministic: bool = bool(get_config(tcfg, "deterministic", False))
        self._seed: Optional[int] = get_config(tcfg, "seed", None)

        self._generator = None
        self._worker_init_fn = None
        if self._deterministic and self._seed is not None:
            g = torch.Generator()
            g.manual_seed(int(self._seed))
            self._generator = g
            def _init_fn(worker_id):
                s = int(self._seed) + worker_id
                random.seed(s); np.random.seed(s); torch.manual_seed(s)
            self._worker_init_fn = _init_fn


    def _normalize_split(self, split: str) -> str:
        s = self._ALIASES.get((split or "").strip().lower(), split)
        if s not in self._ALLOWED: raise ValueError(f"Unsupported split '{split}'. Allowed: {sorted(self._ALLOWED)}")
        return s

    def get_dataset(self, split: str, **overrides) -> Dataset:
        if overrides: return self.build_dataset(split, **overrides)
        if split not in self._datasets: self._datasets[split] = self.build_dataset(split)
        return self._datasets[split]

    def get_loader(self, split: str, **overrides) -> DataLoader:
        split = self._normalize_split(split)
        if overrides:
            # --- split overrides ---
            dataset_overrides = {k: v for k, v in overrides.items()
                                 if k not in self._LOADER_ARG_KEYS}
            loader_overrides  = {k: v for k, v in overrides.items()
                                 if k in self._LOADER_ARG_KEYS and v is not None}

            ds: Dataset = overrides.get("dataset") or self.build_dataset(split, **dataset_overrides)

            args = self.default_loader_args(split, ds)
            collate_fn = loader_overrides.pop("collate_fn", None)
            args.update(loader_overrides)
            if collate_fn is not None:
                args["collate_fn"] = collate_fn

            return DataLoader(ds, **args)

        if split not in self._loaders:
            ds = self.get_dataset(split)
            args = self.default_loader_args(split, ds)
            self._loaders[split] = DataLoader(ds, **args)
        return self._loaders[split]

    def default_loader_args(self, split: str, dataset: Dataset) -> Dict[str, Any]:
        split = self._normalize_split(split)
        is_train = (split == "train")
        batch_size = self.batch_size if is_train else self.eval_batch_size
        args: Dict[str, Any] = dict(
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=is_train,
            persistent_workers=(self.num_workers > 0 and self.persistent_workers),
            timeout=self.timeout,
            generator=self._generator,
            worker_init_fn=self._worker_init_fn,
        )
        if self.prefetch_factor is not None and self.num_workers > 0:
            args["prefetch_factor"] = int(self.prefetch_factor)
        return args

    @abstractmethod
    def build_dataset(self, split: str, **overrides) -> Dataset: ...

class BaseUEBuilder(BaseDatasetBuilder):
    """
    General UE builder:
      - train: ConcatDataset(UEKey(train_clean), UEKey(val_clean))
      - val  : None (no online validation during UE training phase)
      - test : Reuse the clean test from the base task builder
    By default, the transforms for the UE phase are overridden as:
      - train: normalize=False, geom_aug=False (keep Resize/ToTensor, do not Normalize or apply geometric augmentation)
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._base_builder_name = str(get_config(config, "ue.base_task_builder", "mimic_cxr"))

    def _base(self):
        return get_dataset_builder(self._base_builder_name)(self.config)


    def _key_spec(self) -> DictConfig:
        return require_config(self.config, "ue.key", type_=DictConfig)

    def _ue_default_overrides(self, split: str) -> Dict[str, Any]:
        """
        Default transform overrides for the UE phase; if the caller explicitly provides transform/normalize/geom_aug, those take precedence.
        - Training: do not Normalize (because we add noise to x and then Normalize inside the wrapper/trainer), do not apply geometric augmentation
        - Others: keep the base builder defaults (do not force here)
        """
        split = split.lower()
        if split == "train":
            return {"normalize": False, "geom_aug": False}
        return {}

    def _merge_overrides(self, user: Dict[str, Any], ue_defaults: Dict[str, Any]) -> Dict[str, Any]:
        # Explicitly provided transform/normalize/geom_aug by the user take precedence
        out = dict(ue_defaults)
        for k, v in (user or {}).items():
            out[k] = v
        return out

    def build_dataset(self, split: str, **overrides) -> Optional[Dataset]:
        split = self._normalize_split(split)
        base = self._base()

        # Merge UE default overrides
        merged = self._merge_overrides(overrides, self._ue_default_overrides(split))

        if split == "train":
            ds_tr = base.get_dataset("train", **merged)  # clean
            ds_va = base.get_dataset("val",   **merged)  # clean

            # Delayed import to avoid circular dependency
            from .uekey_dataset import UEKeyDataset

            key_spec = self._key_spec()
            ds_tr_k = UEKeyDataset(ds_tr, key_spec)
            ds_va_k = UEKeyDataset(ds_va, key_spec)
            return UEConcatDataset([ds_tr_k, ds_va_k])

        if split == "val":
            return None

        if split == "test":
            return base.get_dataset("test", **merged)

        raise ValueError(f"Unsupported split '{split}' for UE builder.")

    def get_loader(self, split: str, **overrides):
        split = self._normalize_split(split)
        if split == "val":
            return None
        # Also inject UE default overrides into the DataLoader construction path (so the caller can override)
        merged = self._merge_overrides(overrides, self._ue_default_overrides(split))
        return super().get_loader(split, **merged)