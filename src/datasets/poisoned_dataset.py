# file: src/datasets/poisoned_dataset.py
from __future__ import annotations
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import Dataset

from ..core.ue_artifacts import UEShardsAccessor
from ..core.ue_keys import extract_key


def _normalize_inplace(img: torch.Tensor, mean, std):
    """
    In-place per-channel normalize for 3D volume.

    Assumptions:
      - img: [C, D, H, W], float32
      - mean/std: per-channel statistics (len >= C is fine)
    """
    if img.ndim != 4:
        raise ValueError(f"3DUE expects image tensor [C,D,H,W], got {img.shape}")
    C = img.shape[0]
    for c, (m, s) in enumerate(zip(mean, std)):
        if c >= C:
            break
        img[c].sub_(float(m)).div_(float(s))
    return img


class PoisonedDataset(Dataset):
    """
    3D version of PoisonedDataset for segmentation tasks.

    Pipeline:
      1. base.__getitem__(idx, do_transform=False) -> raw volume (in [0,1], not normalized).
      2. Use key_spec to extract key, query UEShardsAccessor for noise [C,D,H,W].
      3. Add noise in [0,1] and clamp.
      4. Call base.transform(image, label) for spatial/data augmentation.
      5. Apply Normalize inside this wrapper.

    """

    def __init__(
        self,
        base: Dataset,
        *,
        perturb_type: str,                 # "classwise" | "samplewise"
        key_spec: Dict[str, Any],          # {"type","from","field",...}
        source_cfg: Dict[str, Any],        # {"type":"files"|"shards"|"manifest", "manifest_path": "..."}
        clamp: Tuple[float, float] = (0.0, 1.0),
        apply_stage: str = "before_normalize",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.base = base
        self.perturb_type = str(perturb_type)
        self.key_spec = dict(key_spec or {})
        self.clamp_min, self.clamp_max = map(float, clamp)
        self.apply_stage = str(apply_stage)
        self.mean = tuple(mean)
        self.std = tuple(std)

        self.transform = getattr(base, "transform", None)

        stype = str(source_cfg.get("type", "files")).lower()
        if stype not in ("files", "shards", "manifest"):
            raise ValueError(
                f"[PoisonedDataset] Only source.types in {{'files','shards','manifest'}} "
                f"are supported, got {stype!r}"
            )

        manifest_path = source_cfg.get("manifest_path", None)
        if manifest_path is None:
            raise ValueError(
                "[PoisonedDataset] source_cfg.manifest_path is required for offline UE."
            )

        self.accessor = UEShardsAccessor.from_manifest(manifest_path)

        if self.apply_stage != "before_normalize":
            raise ValueError(
                "PoisonedDataset currently requires apply_stage='before_normalize' for 3DUE."
            )

    # ---------- attribute delegation ----------
    def __getattr__(self, name: str):
        """
        Delegate unknown attributes to the base dataset
        (e.g., task_type, spacing, metadata).
        """
        return getattr(self.base, name)

    # ---------- dataset protocol ----------
    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dict sample for 3D segmentation:
          - image: [C,D,H,W] float32, with noise & Normalize applied.
          - label: [D,H,W]   long.
          - other fields are forwarded from base.
        """
        # 1. Raw sample (no transform)
        try:
            sample = self.base.__getitem__(idx, do_transform=False)
        except TypeError:
            # Fallback if base does not support do_transform flag
            sample = self.base[idx]

        img: torch.Tensor = sample["image"].float()  # [C,D,H,W]
        if img.ndim != 4:
            raise ValueError(f"3DUE expects image tensor [C,D,H,W], got {img.shape}")

        # 2. Key & noise
        ktype_cfg = str(self.key_spec.get("type", "samplewise"))
        _, key = ktype_cfg, extract_key(sample, idx, self.key_spec)
        if ktype_cfg != self.perturb_type:
            raise ValueError(
                f"perturb_type mismatch: wrapper={self.perturb_type}, "
                f"key_spec.type={ktype_cfg}"
            )

        noise = self.accessor.get(key, perturb_type=self.perturb_type)  # [C,D,H,W] or [1,D,H,W]/[D,H,W]
        noise = noise.to(torch.float32)

        if noise.ndim == 3:
            # [D,H,W] -> [1,D,H,W]
            noise = noise.unsqueeze(0)
        if noise.ndim != 4:
            raise ValueError(
                f"3DUE expects noise tensor [C,D,H,W] or [D,H,W], got {noise.shape}"
            )

        if noise.shape[0] == 1 and img.shape[0] > 1:
            noise = noise.repeat(img.shape[0], 1, 1, 1)

        if noise.shape != img.shape:
            raise ValueError(
                f"noise shape mismatch: noise={tuple(noise.shape)}, img={tuple(img.shape)}"
            )

        if noise.device != img.device:
            noise = noise.to(img.device)

        # 3. Add noise in [0,1] and clamp
        img = torch.clamp(img + noise, min=self.clamp_min, max=self.clamp_max)

        # 4. Spatial/data transforms
        if self.transform is None:
            raise RuntimeError(
                "transform is required for 3D segmentation in PoisonedDataset."
            )

        label = sample.get("label", None)
        if label is None:
            raise RuntimeError("3D segmentation sample must contain 'label' key.")

        img_t, label_t = self.transform(img, label)
        sample["label"] = label_t.long()

        # 5. Normalize
        if self.apply_stage == "before_normalize":
            _normalize_inplace(img_t, self.mean, self.std)

        sample["image"] = img_t
        return sample
