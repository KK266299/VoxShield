# file: src/datasets/flare21.py
from __future__ import annotations

import os
from typing import Optional, Callable, Any, List, Union, Dict

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from omegaconf import DictConfig

from ..utils.logger import get_logger
from ..utils.config import require_config, get_config
from ..registry import register_dataset_builder
from .base_builder import BaseDatasetBuilder, BaseUEBuilder
from .transforms import get_seg_transforms


# ======================================================================
#   FLARE21 3D Volume Dataset
# ======================================================================

class FLARE21VolumeDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        grades: Optional[Union[str, List[str]]] = None,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Any]] = None,
        logger=None,
    ):
        super().__init__()
        self.logger = logger or get_logger()
        self.csv_path = csv_path
        self.split = str(split).lower()
        self.transform = transform

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[FLARE21] CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required_cols = ["case_id", "grade", "volume_path"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"[FLARE21] CSV missing required column: {c}")

        if grades is not None:
            if isinstance(grades, str):
                grades = [grades]
            grades_upper = [g.upper() for g in grades]
            df["grade"] = df["grade"].astype(str).str.upper()
            df = df[df["grade"].isin(grades_upper)].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"[FLARE21] No samples in CSV after filtering: "
                f"csv_path={csv_path}, split={self.split}, grades={grades}"
            )

        self.df = df.reset_index(drop=True)

        self.logger.info(
            f"[FLARE21] Loaded split='{self.split}' from {csv_path}: "
            f"{len(self.df)} cases"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        h5_path = row["volume_path"]

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"[FLARE21] h5 file not found: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            image_np = f["image"][()]  # (C, H, W, D), float32, [0,1], C=1 for CT
            label_np = f["label"][()]  # (H, W, D), uint8
            case_id_attr = f.attrs.get("case_id", None)

        if image_np.ndim != 4:
            raise ValueError(f"[FLARE21] image ndim={image_np.ndim}, expected 4 (C,H,W,D)")
        if label_np.ndim != 3:
            raise ValueError(f"[FLARE21] label ndim={label_np.ndim}, expected 3 (H,W,D)")

        image = torch.from_numpy(image_np).float().permute(0, 3, 1, 2)  # [C,D,H,W]
        label = torch.from_numpy(label_np.astype(np.int64)).long().permute(2, 0, 1)  # [D,H,W]

        if self.transform is not None:
            out = self.transform(image, label)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                image, label = out
            else:
                raise RuntimeError(
                    "[FLARE21] transform must return (image, label), "
                    f"got type={type(out)}"
                )

        case_id = str(row["case_id"])
        grade = str(row["grade"])

        if case_id_attr is not None and str(case_id_attr) != case_id:
            self.logger.warning(
                f"[FLARE21] case_id mismatch: CSV={case_id}, h5.attr={case_id_attr}"
            )

        return {
            "image": image,       # [C,D,H,W], C=1 for CT
            "label": label,       # [D,H,W]
            "case_id": case_id,
            "grade": grade,
            "index": int(idx),    # for UE noise indexing
            "h5_path": h5_path,
        }

class Flare21Builder(BaseDatasetBuilder):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        dcfg: DictConfig = require_config(config, "dataset")

        train_csv = require_config(dcfg, "train_csv_path", type_=str)
        val_csv   = require_config(dcfg, "val_csv_path", type_=str)
        test_csv  = require_config(dcfg, "test_csv_path", type_=str)
        self.csv_paths = {
            "train": train_csv,
            "val":   val_csv,
            "test":  test_csv,
        }

        self.grades = get_config(dcfg, "grades", None)

    def build_dataset(self, split: str, **overrides) -> Dataset:
        split_norm = self._normalize_split(split)

        csv_path = overrides.get("csv_path", self.csv_paths.get(split_norm))
        if csv_path is None:
            raise ValueError(f"[FLARE21] No CSV path configured for split '{split_norm}'.")

        grades = overrides.get("grades", self.grades)

        transform = overrides.get("transform", None)

        if transform is None:
            dcfg: DictConfig = require_config(self.config, "training.data")
            tcfg: DictConfig = get_config(dcfg, "transforms", DictConfig({}))

            normalize = bool(require_config(tcfg, "normalize"))
            geom_aug = bool(require_config(tcfg, "geom_aug"))
            intensity_aug = bool(require_config(tcfg, "intensity_aug"))
            mean = get_config(tcfg, "mean", [0.0])
            std = get_config(tcfg, "std", [1.0])

            transform = get_seg_transforms(
                ndim=3,
                split=split_norm,
                normalize=normalize,
                geom_aug=geom_aug,
                intensity_aug=intensity_aug,
                mean=mean,
                std=std,
            )

        ds = FLARE21VolumeDataset(
            csv_path=csv_path,
            split=split_norm,
            grades=grades,
            transform=transform,
            logger=self.logger,
        )
        return ds

@register_dataset_builder("flare21_seg")
class Flare21SegBuilder(Flare21Builder):

    def __init__(self, config: DictConfig):
        super().__init__(config)


@register_dataset_builder("flare21_ue")
class Flare21UEBuilder(BaseUEBuilder):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._base_builder_name = "flare21_seg"