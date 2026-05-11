# file: src/evaluation/flare21_eval.py
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


@register_evaluation_strategy("flare21_seg")
class Flare21SegmentationEvaluationStrategy:
    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or DictConfig({})

        seg_cfg = get_config(self.config, "evaluation.seg", DictConfig({}))
        ci = get_config(seg_cfg, "class_indices", DictConfig({}))

        self.idx_bg       = int(get_config(ci, "bg",       0))
        self.idx_liver    = int(get_config(ci, "liver",    1))
        self.idx_kidney   = int(get_config(ci, "kidney",   2))
        self.idx_spleen   = int(get_config(ci, "spleen",   3))
        self.idx_pancreas = int(get_config(ci, "pancreas", 4))

        # MONAI metrics on [B, 4, D, H, W] for (Liver, Kidney, Spleen, Pancreas)
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

        # Optional loss for reporting (should align with training config)
        train_crit_cfg = get_config(self.config, "training.criterion", DictConfig({}))
        loss_cfg = get_config(self.config, "evaluation.loss", DictConfig({}))

        include_background = bool(get_config(loss_cfg, "include_background",
                                             get_config(train_crit_cfg, "include_background", False)))
        squared_pred = bool(get_config(loss_cfg, "squared_pred",
                                       get_config(train_crit_cfg, "squared_pred", False)))
        jaccard = bool(get_config(loss_cfg, "jaccard",
                                  get_config(train_crit_cfg, "jaccard", False)))
        lambda_dice = float(get_config(loss_cfg, "lambda_dice",
                                       get_config(train_crit_cfg, "lambda_dice", 1.0)))
        lambda_ce = float(get_config(loss_cfg, "lambda_ce",
                                     get_config(train_crit_cfg, "lambda_ce", 1.0)))
        ce_weight = get_config(loss_cfg, "ce_weight",
                              get_config(train_crit_cfg, "ce_weight", None))
        if ce_weight is not None:
            self._ce_weight_list = ce_weight
        else:
            self._ce_weight_list = None

        # 5-class multi-class Dice+CE loss (weight will be set in evaluate_epoch)
        self.loss_fn_config = {
            "include_background": include_background,
            "to_onehot_y": True,
            "softmax": True,
            "squared_pred": squared_pred,
            "jaccard": jaccard,
            "lambda_dice": lambda_dice,
            "lambda_ce": lambda_ce,
            "reduction": "mean",
        }
        self.loss_fn = None  

    # ------------------------------------------------------------------ #
    # helpers: build 4 FLARE21 organ masks from label id map
    # ------------------------------------------------------------------ #

    def _build_region_masks(self, y_id: torch.Tensor) -> torch.Tensor:

        liver    = self.idx_liver
        kidney   = self.idx_kidney
        spleen   = self.idx_spleen
        pancreas = self.idx_pancreas

        # Individual organ regions
        y_liver    = (y_id == liver)
        y_kidney   = (y_id == kidney)
        y_spleen   = (y_id == spleen)
        y_pancreas = (y_id == pancreas)

        y_reg = torch.stack(
            [y_liver.float(), y_kidney.float(), y_spleen.float(), y_pancreas.float()],
            dim=1,   # -> [B, 4, D, H, W]
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

        if self.loss_fn is None:
            weight = None
            if self._ce_weight_list is not None:
                weight = torch.tensor(self._ce_weight_list, dtype=torch.float32, device=device)
            self.loss_fn_config["weight"] = weight
            self.loss_fn = DiceCELoss(**self.loss_fn_config)

        total_loss = 0.0
        n_samples = 0

        # reset accumulators
        self.dice_metric.reset()
        self.miou_metric.reset()

        pbar = tqdm(data_loader, desc="Evaluate SEG (FLARE21)", leave=False)
        for batch in pbar:
            x = batch["image"].to(device)                # [B, C, D, H, W]
            y_raw = batch["label"].to(device).long()     # [B,D,H,W] or [B,1,D,H,W]

            if y_raw.ndim == 5:
                if y_raw.size(1) != 1:
                    raise ValueError(f"[Flare21SegEval] label ndim=5 but channel={y_raw.size(1)} != 1")
                y_id = y_raw[:, 0]
            elif y_raw.ndim == 4:
                y_id = y_raw
            else:
                raise ValueError(f"[Flare21SegEval] Unsupported label shape: {y_raw.shape}")

            # --- build FLARE21 region GT: [B,4,D,H,W] (Liver, Kidney, Spleen, Pancreas) ---
            y_reg = self._build_region_masks(y_id)

            # --- forward ---
            logits = model(x)                            # [B, 5, D, H, W]

            # multi-class prediction
            prob = torch.softmax(logits, dim=1)          # [B, 5, D, H, W]
            y_pred_id = prob.argmax(dim=1)               # [B, D, H, W]

            # --- build FLARE21 region prediction ---
            y_pred_reg = self._build_region_masks(y_pred_id)  # [B,4,D,H,W]

            # --- accumulate metrics (region-based) ---
            self.dice_metric(y_pred=y_pred_reg, y=y_reg)
            self.miou_metric(y_pred=y_pred_reg, y=y_reg)

            # --- val loss（5-class multi-class DiceCE）---
            loss = self.loss_fn(logits, y_id.unsqueeze(1))
            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            n_samples += bs

        # ---- aggregate Dice with not_nans ----
        dice, not_nans = self.dice_metric.aggregate()
        if dice.ndim == 1:
            dice = dice.view(1, -1)
            not_nans = not_nans.view(1, -1)
        elif dice.ndim == 2:
            pass
        else:
            dice = dice.view(-1, 4)
            not_nans = not_nans.view(-1, 4)

        region_dice = []
        region_has_samples = []
        region_names = ["liver", "kidney", "spleen", "pancreas"]

        for c in range(4):
            val_mask = not_nans[:, c] > 0
            has_samples = bool(val_mask.any().item())
            region_has_samples.append(has_samples)

            if has_samples:
                mean_c = dice[val_mask, c].mean()
                region_dice.append(float(mean_c.item()))
            else:
                region_dice.append(0.0)

        liver_dc, kidney_dc, spleen_dc, pancreas_dc = region_dice

        if any(region_has_samples):
            valid_vals = [
                d for d, flag in zip(region_dice, region_has_samples) if flag
            ]
            avg_dc = float(sum(valid_vals) / len(valid_vals))
        else:
            avg_dc = 0.0

        # ---- aggregate IoU with not_nans ----
        miou_vals, miou_not_nans = self.miou_metric.aggregate()
        if miou_vals.ndim == 1:
            miou_vals = miou_vals.view(1, -1)
            miou_not_nans = miou_not_nans.view(1, -1)
        elif miou_vals.ndim == 2:
            pass
        else:
            miou_vals = miou_vals.view(-1, 4)
            miou_not_nans = miou_not_nans.view(-1, 4)

        region_iou = []
        region_has_iou_samples = []

        for c in range(4):
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
            "loss":        float(total_loss / max(1, n_samples)),
            "liver_dc":    liver_dc,
            "kidney_dc":   kidney_dc,
            "spleen_dc":   spleen_dc,
            "pancreas_dc": pancreas_dc,
            "avg_dc":      avg_dc,
            "miou":        miou,
            "jc":          miou,   # alias
        }

        # reset for next epoch call
        self.dice_metric.reset()
        self.miou_metric.reset()

        return metrics