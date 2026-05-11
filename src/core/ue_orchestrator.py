# file: src/core/ue_orchestrator.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ..utils.config import get_config, require_config
from ..registry import PROVIDERS
from ..datasets.poisoned_dataset import PoisonedDataset
from .ue_artifacts import write_shards, write_files
from .ue_keys import collect_keys  # Unified key extraction and collection interface

def build_unlearnable_provider_instance(
    config: DictConfig,
    train_dataset,
    val_dataset=None,
) -> Optional[object]:
    pcfg = get_config(config, "training.data.poison", OmegaConf.create({}))
    if not bool(get_config(pcfg, "enabled", False)):
        return None

    stype = str(require_config(config, "training.data.poison.source.type")).lower()
    if stype != "provider":
        return None

    tcfg = require_config(config, "training.data.transforms")
    src = get_config(pcfg, "source", OmegaConf.create({}))
    prov = get_config(src, "provider", OmegaConf.create({}))

    name = str(get_config(prov, "name", "")).lower()
    params = dict(get_config(prov, "params", OmegaConf.create({})))

    if "image_size" not in params:
        params["image_size"] = tuple(require_config(tcfg, "image_size"))

    ProviderCls = PROVIDERS.get(name)
    if ProviderCls is None:
        raise ValueError(f"[UE] Unknown provider: {name!r}")

    requires_keys = bool(getattr(ProviderCls, "REQUIRES_KEYS_AT_INIT", False))

    if requires_keys:
        key_spec = get_config(
            config,
            "ue.key",
            OmegaConf.create({"type": "samplewise", "from": "index"}),
        )
        ktype = str(get_config(key_spec, "type", "samplewise")).lower()
        classwise = ktype == "classwise"

        union: List[Any] = []
        seen = set()
        for ds in (train_dataset, val_dataset):
            if ds is None:
                continue
            ks = collect_keys(ds, key_spec, classwise=classwise)
            if classwise:
                for k in ks:
                    if k not in seen:
                        union.append(k)
                        seen.add(k)
            else:
                union.extend(ks)

        params = dict(params)
        params["keys"] = union 

    provider_instance = ProviderCls(**params)
    return provider_instance


def attach_unlearnable_noise(
    config: DictConfig,
    dataset,
    *,
    provider_instance: Optional[object] = None,
):
    pcfg = get_config(config, "training.data.poison", OmegaConf.create({}))
    if not bool(get_config(pcfg, "enabled", False)):
        return dataset

    if provider_instance is not None:
        print(
            "[UE] Note: provider_instance is ignored in attach_unlearnable_noise; "
            "3D pipeline only reads noise from offline manifest."
        )

    tcfg = get_config(config, "training.data.transforms", OmegaConf.create({}))
    key_spec = require_config(pcfg, "key")
    perturb_type = str(require_config(pcfg, "perturb_type"))

    mean = tuple(get_config(tcfg, "mean", [0.0, 0.0, 0.0]))
    std = tuple(get_config(tcfg, "std", [1.0, 1.0, 1.0]))

    src_cfg = require_config(pcfg, "source")
    stype = str(get_config(src_cfg, "type", "files")).lower()
    if stype not in {"files", "shards", "manifest"}:
        raise ValueError(
            f"[UE] 3D pipeline only supports poison.source.type in "
            f"{{'files','shards','manifest'}}, got {stype!r}"
        )

    manifest_path = require_config(src_cfg, "manifest_path", type_=str)

    clamp_min = float(get_config(pcfg, "clamp_min", 0.0))
    clamp_max = float(get_config(pcfg, "clamp_max", 1.0))
    apply_stage = str(get_config(pcfg, "apply_stage", "before_normalize"))

    # defense augmentation config (optional, for ablation experiments)
    defense_cfg = get_config(pcfg, "defense", None)
    if defense_cfg is not None:
        defense_cfg = OmegaConf.to_container(defense_cfg, resolve=True)

    return PoisonedDataset(
        base=dataset,
        perturb_type=perturb_type,
        key_spec=key_spec,
        source_cfg={"type": stype, "manifest_path": manifest_path},
        clamp=(clamp_min, clamp_max),
        apply_stage=apply_stage,
        mean=mean,
        std=std,
        defense_cfg=defense_cfg,
    )

def generate_training_free(
    config: DictConfig,
    datasets: Union[torch.utils.data.Dataset, Sequence[torch.utils.data.Dataset]],
) -> bool:

    ue_cfg = get_config(config, "ue", OmegaConf.create({}))
    alg = get_config(ue_cfg, "algorithm", OmegaConf.create({}))
    if str(get_config(alg, "kind", "")).lower() != "training_free":
        return False

    prov_name = str(get_config(alg, "name", "")).lower()
    prov_cls = PROVIDERS.get(prov_name)
    if prov_cls is None:
        raise ValueError(f"[UE] Unknown training-free provider: {prov_name!r}")

    params = dict(get_config(alg, "params", OmegaConf.create({})))
    epsilon = float(get_config(alg, "params.epsilon", params.get("epsilon", 0.0313725)))

    ds_list = list(datasets) if isinstance(datasets, (list, tuple)) else [datasets]

    key_spec = require_config(ue_cfg, "key")
    perturb_type = str(require_config(key_spec, "type"))
    classwise = perturb_type.lower() == "classwise"

    union: List[Any] = []
    seen = set()
    for ds in ds_list:
        if ds is None:
            continue
        ks = collect_keys(ds, key_spec, classwise=classwise)
        if classwise:
            for k in ks:
                if k not in seen:
                    union.append(k)
                    seen.add(k)
        else:
            union.extend(ks)

    requires_keys = bool(getattr(prov_cls, "REQUIRES_KEYS_AT_INIT", False))
    prov_params = dict(params)
    if requires_keys:
        prov_params["keys"] = union

    provider = prov_cls(**prov_params)

    entries: List[Tuple[Any, torch.Tensor]] = []
    for k in tqdm(union, desc="UE gen (training-free 3D)", unit="key"):
        n = provider.get_noise(k, perturb_type)
        if not isinstance(n, torch.Tensor):
            raise TypeError(
                f"[UE] provider.get_noise must return torch.Tensor, got {type(n)}"
            )

        if n.ndim != 4:
            raise ValueError(
                f"[UE] 3D training-free UE expects noise shape [C,D,H,W], "
                f"got {tuple(n.shape)} for key={k!r}"
            )
        entries.append((k, n))

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
        raise ValueError(f"[UE] Unknown ue.io.strategy: {strategy!r}")

    if not os.path.exists(manifest_path):
        raise RuntimeError("[UE] training-free generation failed to write manifest.")

    print(f"[UE] training-free 3D manifest saved to: {manifest_path}")
    return True