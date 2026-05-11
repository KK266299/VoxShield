# file: src/evaluation/brats19_seg.py
from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from monai.metrics import DiceMetric, MeanIoU
from monai.losses import DiceCELoss
from tqdm import tqdm

from ..utils.config import get_config
from ..registry import register_evaluation_strategy


@register_evaluation_strategy("brats19_seg")
class Brats19SegmentationEvaluationStrategy:
    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or DictConfig({})

        seg_cfg = get_config(self.config, "evaluation.seg", DictConfig({}))
        ci = get_config(seg_cfg, "class_indices", DictConfig({}))

        self.idx_bg    = int(get_config(ci, "bg",    0))
        self.idx_ncr   = int(get_config(ci, "ncr",   1))
        self.idx_edema = int(get_config(ci, "edema", 2))
        self.idx_enh   = int(get_config(ci, "enh",   3))

        self.dice_metric = DiceMetric(
            include_background=True,
            reduction="none",          
            get_not_nans=True,        
        )
        self.miou_metric = MeanIoU(
            include_background=True,
            reduction="none",
            get_not_nans=True,
        )

        # Optional loss for reporting (align with training if desired)
        loss_cfg = get_config(self.config, "evaluation.loss", DictConfig({}))
        include_background = bool(get_config(loss_cfg, "include_background", False))
        squared_pred = bool(get_config(loss_cfg, "squared_pred", False))
        jaccard = bool(get_config(loss_cfg, "jaccard", False))
        lambda_dice = float(get_config(loss_cfg, "lambda_dice", 1.0))
        lambda_ce = float(get_config(loss_cfg, "lambda_ce", 1.0))

        # logits: [B,4,D,H,W]
        self.loss_fn = DiceCELoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
            squared_pred=squared_pred,
            jaccard=jaccard,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            reduction="mean",
        )

    # ------------------------------------------------------------------ #
    # helpers: build 3 BraTS regions from label id map
    # ------------------------------------------------------------------ #

    def _build_region_masks(self, y_id: torch.Tensor) -> torch.Tensor:
        bg    = self.idx_bg
        ncr   = self.idx_ncr
        edema = self.idx_edema
        enh   = self.idx_enh

        # enhancing tumour (ET)
        y_et = y_id.eq(enh)

        # tumour core (TC): NCR/NET + Enhancing
        y_tc = y_id.eq(ncr) | y_id.eq(enh)

        # whole tumour (WT): all non-background
        y_wt = y_id.ne(bg)

        y_reg = torch.stack(
            [y_et.float(), y_tc.float(), y_wt.float()],
            dim=1,   # -> [B, 3, D, H, W]
        )
        return y_reg

    @torch.no_grad()
    def evaluate_epoch(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        model.eval()
        model.to(device)

        total_loss = 0.0
        n_samples = 0

        # reset accumulators
        self.dice_metric.reset()
        self.miou_metric.reset()

        pbar = tqdm(data_loader, desc="Evaluate SEG (BraTS19)", leave=False)
        for batch in pbar:
            x = batch["image"].to(device)                # [B, C, D, H, W]
            y_raw = batch["label"].to(device).long()     #  [B,D,H,W]  [B,1,D,H,W]

            #  [B,D,H,W]
            if y_raw.ndim == 5:
                # [B,1,D,H,W] -> [B,D,H,W]
                if y_raw.size(1) != 1:
                    raise ValueError(f"[Brats19SegEval] label ndim=5 but channel={y_raw.size(1)} != 1")
                y_id = y_raw[:, 0]
            elif y_raw.ndim == 4:
                y_id = y_raw
            else:
                raise ValueError(f"[Brats19SegEval] Unsupported label shape: {y_raw.shape}")

            # --- build BraTS region GT: [B,3,D,H,W] (ET,TC,WT) ---
            y_reg = self._build_region_masks(y_id)

            # --- forward ---
            logits = model(x)                            # [B, 4, D, H, W]

            # multi-class prediction
            prob = torch.softmax(logits, dim=1)          # [B, 4, D, H, W]
            y_pred_id = prob.argmax(dim=1)               # [B, D, H, W]

            # --- build BraTS region prediction ---
            y_pred_reg = self._build_region_masks(y_pred_id)  # [B,3,D,H,W]

            # --- accumulate metrics (region-based) ---
            self.dice_metric(y_pred=y_pred_reg, y=y_reg)
            self.miou_metric(y_pred=y_pred_reg, y=y_reg)

            loss = self.loss_fn(logits, y_id.unsqueeze(1))
            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            n_samples += bs

        # ---- aggregate Dice with not_nans ----
        dice, not_nans = self.dice_metric.aggregate()
        dice = dice.view(-1, 3)
        not_nans = not_nans.view(-1, 3)

        region_dice = []
        region_has_samples = []

        for c in range(3):  # 0:ET, 1:TC, 2:WT
            val_mask = not_nans[:, c] > 0   
            has_samples = bool(val_mask.any().item())
            region_has_samples.append(has_samples)

            if has_samples:
                mean_c = dice[val_mask, c].mean()
                region_dice.append(float(mean_c.item()))
            else:
                region_dice.append(0.0)

        et_dc, tc_dc, wt_dc = region_dice

        if any(region_has_samples):
            valid_vals = [
                d for d, flag in zip(region_dice, region_has_samples) if flag
            ]
            avg_dc = float(sum(valid_vals) / len(valid_vals))
        else:
            avg_dc = 0.0

        miou_vals, miou_not_nans = self.miou_metric.aggregate()
        miou_vals = miou_vals.view(-1, 3)
        miou_not_nans = miou_not_nans.view(-1, 3)

        region_iou = []
        region_has_iou_samples = []

        for c in range(3):
            val_mask = miou_not_nans[:, c] > 0
            has_samples = bool(val_mask.any().item())
            region_has_iou_samples.append(has_samples)

            if has_samples:
                mean_c = miou_vals[val_mask, c].mean()
                region_iou.append(float(mean_c.item()))
            else:
                region_iou.append(0.0)

        if any(region_has_iou_samples):
            valid_iou_vals = [
                v for v, flag in zip(region_iou, region_has_iou_samples) if flag
            ]
            miou = float(sum(valid_iou_vals) / len(valid_iou_vals))
        else:
            miou = 0.0

        metrics = {
            "loss":   float(total_loss / max(1, n_samples)),
            "et_dc":  et_dc,
            "tc_dc":  tc_dc,
            "wt_dc":  wt_dc,
            "avg_dc": avg_dc,
            "miou":   miou,
            "jc":     miou,   # alias
        }

        # reset for next epoch call
        self.dice_metric.reset()
        self.miou_metric.reset()

        return metrics
