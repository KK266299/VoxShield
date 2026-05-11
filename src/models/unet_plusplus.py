# src/models/unet_plusplus.py
"""
U-Net++ (Nested U-Net) Implementation based on MONAI

Reference:
    Zhou, Z., et al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
    DLMIA 2018. https://arxiv.org/abs/1807.10165
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Sequence, Union
from omegaconf import DictConfig, OmegaConf
from monai.networks.nets import BasicUNetPlusPlus

from src.utils.logger import get_logger
from src.utils.config import get_config
from src.registry import register_model


@register_model("unet_plusplus")
class UNetPlusPlus(BasicUNetPlusPlus):
    def __init__(
        self,
        cfg: DictConfig | Dict[str, Any],
        in_channels: Optional[int] = None,
        eps: Optional[float] = None,
    ):
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        log = get_logger()

        c_in_cfg = get_config(cfg, "in_channels", 3)
        c_in = (
            in_channels
            if in_channels is not None
            else (None if c_in_cfg == "auto" else int(c_in_cfg))
        )
        if c_in is None:
            raise ValueError(
                "[UNetPlusPlus] in_channels is 'auto'; please pass in_channels at construction time."
            )

        out_ch = int(get_config(cfg, "num_classes", 1))

        channels = list(get_config(cfg, "channels", [32, 64, 128, 256, 512]))
        spatial_dims = int(get_config(cfg, "spatial_dims", 3))
        act = get_config(cfg, "act", "relu")
        norm = get_config(cfg, "norm", "BATCH")
        dropout = float(get_config(cfg, "dropout", 0.0))

        # features_list = 32, 32, 64, 128, 256, 32
        features = (32, 32, 64, 128, 256, 32)

        deep_supervision = bool(get_config(cfg, "deep_supervision", False))

        log.info(
            f"[Gen] UNet++: spatial_dims={spatial_dims}, in={c_in}, out={out_ch}, "
            f"features={features}, deep_supervision={deep_supervision}, "
            f"act={act}, norm={norm}, dropout={dropout}"
        )

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=c_in,
            out_channels=out_ch,
            features=features,
            act=act,
            norm=norm,
            dropout=dropout,
            deep_supervision=deep_supervision,
        )

        self._deep_supervision = deep_supervision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        # BasicUNetPlusPlus returns list when deep_supervision=True
        if isinstance(out, (list, tuple)):
            if self._deep_supervision:
                return sum(out) / len(out)
            else:
                return out[-1]
        return out