# file: src/core/ue_orchestrator_roi.py
"""
ROI-based UE orchestrator for Medical Image Segmentation.

This module extends the standard orchestrator to support ROI-based LSP
noise generation, where noise patterns are selected based on segmentation masks.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ..utils.config import get_config, require_config
from ..registry import PROVIDERS
from .ue_artifacts import write_shards, write_files
from .ue_keys import collect_keys


def generate_training_free_roi(
    config: DictConfig,
    datasets: Union[torch.utils.data.Dataset, Sequence[torch.utils.data.Dataset]],
) -> bool:
    """
    ROI-based training-free UE generation for 3D MIS (e.g., BraTS).

    This function generates noise using ROI-aware LSP providers (e.g., lsp_roi)
    that blend background and foreground noise patterns based on segmentation masks.

    Config example:

      ue:
        algorithm:
          kind: training_free
          name: lsp_roi
          params:
            epsilon: 0.0313725
            image_size: [4, 128, 128, 128]  # [C, D, H, W]
            seed: 0
            noise_frame_size: 32
            num_labels: 4
            roi_mode: binary
        key:
          type: samplewise
          from: field
          field: case_id
        store_dir: ./ue/brats19_lsp_roi
        io:
          strategy: files
          dtype: int8

    Behavior:
      1. Collect all keys from datasets (samplewise or classwise)
      2. Initialize ROI-aware provider with all keys
      3. For each sample, get image and label from dataset
      4. Call provider.get_noise_with_mask(key, label) to get ROI-blended noise
      5. Write noise to disk with manifest
    """
    ue_cfg = get_config(config, "ue", OmegaConf.create({}))
    alg = get_config(ue_cfg, "algorithm", OmegaConf.create({}))

    if str(get_config(alg, "kind", "")).lower() != "training_free":
        return False

    prov_name = str(get_config(alg, "name", "")).lower()

    # Only handle ROI-based providers
    if not prov_name.endswith("_roi"):
        return False

    prov_cls = PROVIDERS.get(prov_name)
    if prov_cls is None:
        raise ValueError(f"[UE-ROI] Unknown ROI provider: {prov_name!r}")

    # Provider params
    params = dict(get_config(alg, "params", OmegaConf.create({})))
    epsilon = float(get_config(alg, "params.epsilon", params.get("epsilon", 0.0313725)))

    # Datasets
    ds_list = list(datasets) if isinstance(datasets, (list, tuple)) else [datasets]

    # Key collection
    key_spec = require_config(ue_cfg, "key")
    perturb_type = str(require_config(key_spec, "type"))
    classwise = perturb_type.lower() == "classwise"

    # Collect all unique keys
    union: List[Any] = []
    seen = set()
    for ds in ds_list:
        ks = collect_keys(ds, key_spec, classwise=classwise)
        if classwise:
            for k in ks:
                if k not in seen:
                    union.append(k)
                    seen.add(k)
        else:
            union.extend(ks)

    # Initialize provider
    requires_keys = bool(getattr(prov_cls, "REQUIRES_KEYS_AT_INIT", False))
    prov_params = dict(params)
    if requires_keys:
        prov_params["keys"] = union

    provider = prov_cls(**prov_params)

    # Check if provider supports mask-based noise generation
    if not hasattr(provider, "get_noise_with_mask"):
        raise RuntimeError(
            f"[UE-ROI] Provider '{prov_name}' does not support get_noise_with_mask(). "
            "Please use a ROI-aware provider like 'lsp_roi'."
        )

    # Generate noise with ROI awareness
    # We need to access the original samples to get labels
    # Build a key -> (dataset_idx, sample_idx) mapping

    key_from = str(get_config(key_spec, "from", "index")).lower()
    key_field = str(get_config(key_spec, "field", "case_id"))

    key_to_sample: Dict[Any, Tuple[int, int]] = {}  # key -> (ds_idx, sample_idx)

    for ds_idx, ds in enumerate(ds_list):
        for sample_idx in range(len(ds)):
            sample = ds[sample_idx]

            # Extract key from sample
            if key_from == "index":
                k = sample_idx
            elif key_from == "field":
                if key_field not in sample:
                    raise KeyError(
                        f"[UE-ROI] Sample missing key field '{key_field}': {list(sample.keys())}"
                    )
                k = sample[key_field]
            else:
                raise ValueError(f"[UE-ROI] Unsupported key.from: {key_from}")

            # Canonicalize key
            if torch.is_tensor(k):
                k = k.item() if k.ndim == 0 else tuple(k.cpu().numpy().tolist())

            if classwise:
                # For classwise, only keep first occurrence
                if k not in key_to_sample:
                    key_to_sample[k] = (ds_idx, sample_idx)
            else:
                key_to_sample[k] = (ds_idx, sample_idx)

    # Generate noise for each key
    entries: List[Tuple[Any, torch.Tensor]] = []
    for k in tqdm(union, desc="UE gen (ROI-aware 3D)", unit="key"):
        if k not in key_to_sample:
            raise KeyError(f"[UE-ROI] Key {k!r} not found in datasets")

        ds_idx, sample_idx = key_to_sample[k]
        sample = ds_list[ds_idx][sample_idx]

        # Get label/mask
        if "label" in sample:
            mask = sample["label"]  # [D, H, W]
        elif "mask" in sample:
            mask = sample["mask"]
        else:
            raise KeyError(
                f"[UE-ROI] Sample missing 'label' or 'mask' field: {list(sample.keys())}"
            )

        # Ensure mask is tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)

        # Generate ROI-aware noise
        noise = provider.get_noise_with_mask(k, mask)

        if not isinstance(noise, torch.Tensor):
            raise TypeError(
                f"[UE-ROI] provider.get_noise_with_mask must return torch.Tensor, got {type(noise)}"
            )

        if noise.ndim != 4:
            raise ValueError(
                f"[UE-ROI] 3D noise expects shape [C,D,H,W], got {tuple(noise.shape)} for key={k!r}"
            )

        entries.append((k, noise))

    # Write to disk
    store_dir = str(get_config(ue_cfg, "store_dir", os.path.join(".", "ue")))
    os.makedirs(store_dir, exist_ok=True)

    io_cfg = get_config(ue_cfg, "io", OmegaConf.create({}))
    strategy = str(get_config(io_cfg, "strategy", "files")).lower()

    if strategy == "files":
        manifest_path = write_files(
            store_dir=store_dir,
            entries=entries,
            eps=epsilon,
            perturb_type=perturb_type,
            key_spec=key_spec,
        )
    elif strategy == "shards":
        shard_size = int(get_config(io_cfg, "shard_size", 1000))
        manifest_path = write_shards(
            store_dir=store_dir,
            entries=entries,
            eps=epsilon,
            shard_size=shard_size,
            perturb_type=perturb_type,
            key_spec=key_spec,
        )
    else:
        raise ValueError(f"[UE-ROI] Unknown io.strategy: {strategy!r}")

    if not os.path.exists(manifest_path):
        raise RuntimeError("[UE-ROI] Failed to write manifest.")

    print(f"[UE-ROI] ROI-aware 3D manifest saved to: {manifest_path}")
    return True