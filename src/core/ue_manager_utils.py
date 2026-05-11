# file: src/core/ue_manager_utils.py
from __future__ import annotations
from typing import Any, Dict, Tuple

import os
import torch
from torch.utils.data import Dataset

from ..ue_providers.lsp import LSPProvider
from ..ue_providers.random_noise import RandomNoiseProvider
from ..core.ue_artifacts import save_shards
from ..datasets.poisoned_dataset import PoisonedDataset
from ..utils.config import get_config


def maybe_generate_ue_artifacts(config, train_dataset: Dataset) -> bool:
    """
    If ue.mode=gen_perturb, generate class-wise LSP shards into ue.store_dir and return True.
    Otherwise return False (no-op).
    """
    umode = str(get_config(config, "ue.mode", "train"))
    if umode != "gen_perturb":
        return False

    store_dir = str(get_config(config, "ue.store_dir"))
    gen = get_config(config, "ue.provider", {})
    name = str(get_config(gen, "name", "lsp")).lower()
    if name != "lsp" and name != "random_noise":
        raise ValueError(f"Unsupported UE provider: {name}")

    # image size from training transforms
    tcfg = get_config(config, "training.data.transforms", {})
    C, H, W = tuple(get_config(tcfg, "image_size", [3, 224, 224]))

    if name == "lsp":
        provider = LSPProvider(
            epsilon=float(get_config(gen, "params.epsilon", 0.0313725)),
            image_size=(C, H, W),
            seed=int(get_config(gen, "params.seed", 0)),
            tied_channels=bool(get_config(gen, "params.tied_channels", True)),
            mask=get_config(gen, "params.mask", None),
            per_class=True,
        )
    elif name == "random_noise":
        provider = RandomNoiseProvider(
            epsilon=float(get_config(gen, "params.epsilon", 0.0313725)),
            image_size=(C, H, W),
            seed=int(get_config(gen, "params.seed", 0)),
            mode=str(get_config(gen, "params.mode", "uniform")),
            params=get_config(gen, "params.params", None),
            tied_channels=bool(get_config(gen, "params.tied_channels", True)),
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unsupported UE provider: {name}")

    # Build id order from dataset (expects 'subject_id' field)
    # We'll iterate all class labels in train set
    # Collect unique subject_ids in dataset order
    ids = set()
    id_order = []
    for i in range(len(train_dataset)):
        sid = train_dataset[i]["subject_id"]
        if sid not in ids:
            ids.add(sid)
            id_order.append(sid)

    # Generator yields per-class noise and matching keys
    def _tensor_iter():
        for sid in id_order:
            yield provider.generate_one_class(sid)

    def _keys_iter():
        for sid in id_order:
            yield sid

    # Recommend int8 + scale = eps/127 for compact storage
    eps = float(get_config(gen, "params.epsilon", 0.0313725))
    scale = eps / 127.0
    manifest_path = save_shards(
        store_dir=store_dir,
        tensor_iter=_tensor_iter(),
        keys_iter=_keys_iter(),
        image_size=(C, H, W),
        perturb_type="classwise",
        shard_size=int(get_config(gen, "shard_size", 5000)),
        dtype=str(get_config(gen, "params.dtype", "int8")),
        scale=scale,
        mapping={"id2class": {str(sid): i for i, sid in enumerate(id_order)}},
        manifest_extra={"created_by": name, "epsilon": eps, "norm": "linf"},
    )

    print(f"[UE] {name} manifest saved to: {manifest_path}")
    return True


def wrap_train_with_poison(config, train_dataset: Dataset) -> Dataset:
    """
    If training.data.poison.enabled true, wrap the train dataset with PoisonedDataset.
    """
    pcfg = get_config(config, "training.data.poison", {})
    if not bool(get_config(pcfg, "enabled", False)):
        return train_dataset

    source = get_config(pcfg, "source", {})
    ptype = str(get_config(pcfg, "perturb_type", "classwise"))

    tcfg = get_config(config, "training.data.transforms", {})
    mean = tuple(get_config(tcfg, "mean", [0.485, 0.456, 0.406]))
    std  = tuple(get_config(tcfg, "std",  [0.229, 0.224, 0.225]))

    wrapper = PoisonedDataset(
        base=train_dataset,
        perturb_type=ptype,
        source_cfg=source,
        clamp=(float(get_config(pcfg, "clamp_min", 0.0)), float(get_config(pcfg, "clamp_max", 1.0))),
        apply_stage=str(get_config(pcfg, "apply_stage", "before_normalize")),
        mean=mean,
        std=std,
        io_cache_max_items=int(get_config(pcfg, "io_cache.max_items", 16)),
    )
    return wrapper
