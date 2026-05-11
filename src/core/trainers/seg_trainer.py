# src/trainers/grape_seg.py
from __future__ import annotations
from typing import Dict, Any

import torch
import torch.nn as nn

from ..trainer_base import TrainerBase
from omegaconf import DictConfig
from ...utils.config import get_config
from monai.losses import DiceCELoss


class SegTrainer(TrainerBase):

    def __init__(self, config: DictConfig, device: torch.device, evaluation_strategy):
        super().__init__(config, device)
        self.evaluation_strategy = evaluation_strategy

        # Loss config (mirrors MONAI DiceCELoss signature subset)
        crit_cfg = get_config(config, "training.criterion", DictConfig({}))
        self.include_background = bool(get_config(crit_cfg, "include_background", False))
        self.squared_pred = bool(get_config(crit_cfg, "squared_pred", False))
        self.jaccard = bool(get_config(crit_cfg, "jaccard", False))
        self.lambda_dice = float(get_config(crit_cfg, "lambda_dice", 1.0))
        self.lambda_ce = float(get_config(crit_cfg, "lambda_ce", 1.0))
        self.ce_weight = get_config(crit_cfg, "ce_weight", None)

        self._loss = self._build_loss()

    def _build_loss(self) -> nn.Module:
        weight = None
        if self.ce_weight is not None:
            weight = torch.tensor(self.ce_weight, dtype=torch.float32, device=self.device)

        return DiceCELoss(
            include_background=self.include_background,
            to_onehot_y=True, 
            softmax=True,
            squared_pred=self.squared_pred,
            jaccard=self.jaccard,
            lambda_dice=self.lambda_dice,
            lambda_ce=self.lambda_ce,
            reduction="mean",
            weight=weight,
        )

    def _init_epoch_metrics(self) -> Dict[str, Any]:
        """Initialize metrics for supervised training"""
        from ...utils.metrics import AverageMeter
        return {
            "loss": AverageMeter()
        }

    def _is_best_model(self, eval_stats: Dict[str, float]) -> bool:
        """Determine if current model is best - delegate to evaluation strategy"""
        if hasattr(self.evaluation_strategy, "is_best_model"):
            return self.evaluation_strategy.is_best_model(eval_stats, self.best_metrics)
        # Default judgment based on validation loss
        if eval_stats:
            metric_name = "loss"
            current_val = eval_stats.get(metric_name, 0.0)
            best_val = self.best_metrics.get(metric_name, float("inf"))
            self.logger.info(
                f"Current {metric_name}: {current_val:.4f}, Best {metric_name}: {best_val:.4f}"
            )
            return current_val < best_val
        return False

    def run_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.optimizer.zero_grad()

        x = batch["image"].to(self.device)           # 3D: [B,C,D,H,W]
        y_id = batch["label"].to(self.device).long() # 3D: [B,D,H,W]

        logits = self.model(x)                       # [B,num_classes,D,H,W]
        loss = self._loss(logits, y_id.unsqueeze(1)) # -> y: [B,1,D,H,W]
        loss.backward()
        self.optimizer.step()

        return {"loss": float(loss.item())}