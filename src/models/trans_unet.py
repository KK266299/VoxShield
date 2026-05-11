# src/models/trans_unet.py
"""
UNETR (Transformer-based U-Net) Implementation based on MONAI

Reference:
    Hatamizadeh, A., et al. "UNETR: Transformers for 3D Medical Image Segmentation"
    WACV 2022. https://arxiv.org/abs/2103.10504

Note: MONAI's UNETR is similar to TransUNet, using Vision Transformer as encoder
with U-Net style decoder for medical image segmentation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Sequence, Union
from omegaconf import DictConfig, OmegaConf
from monai.networks.nets import UNETR

from src.utils.logger import get_logger
from src.utils.config import get_config
from src.registry import register_model


@register_model("trans_unet")
class TransUNet(UNETR):
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
                "[TransUNet] in_channels is 'auto'; please pass in_channels at construction time."
            )

        out_ch = int(get_config(cfg, "num_classes", 1))

        spatial_dims = int(get_config(cfg, "spatial_dims", 3))
        norm = get_config(cfg, "norm", "BATCH")
        dropout = float(get_config(cfg, "dropout", 0.0))

        img_size = tuple(get_config(cfg, "img_size", [160, 160, 160] if spatial_dims == 3 else [256, 256]))
        feature_size = int(get_config(cfg, "feature_size", 16))
        hidden_size = int(get_config(cfg, "hidden_size", 768))
        mlp_dim = int(get_config(cfg, "mlp_dim", 3072))
        num_heads = int(get_config(cfg, "num_heads", 12))

        log.info(
            f"[Gen] TransUNet(UNETR): spatial_dims={spatial_dims}, in={c_in}, out={out_ch}, "
            f"img_size={img_size}, feature_size={feature_size}, hidden_size={hidden_size}, "
            f"mlp_dim={mlp_dim}, num_heads={num_heads}, norm={norm}, dropout={dropout}"
        )

        super().__init__(
            in_channels=c_in,
            out_channels=out_ch,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            norm_name=norm,
            dropout_rate=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)