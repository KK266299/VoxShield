# file: src/ue_algos/min_min.py
from __future__ import annotations
from typing import Dict, Iterable, List

import torch
from omegaconf import DictConfig
from monai.losses import DiceCELoss

from ...registry import register_plugin
from ...utils.config import get_config, require_config


@register_plugin("min_min")
class MinMinUE:
    def __init__(self):
        self._seg_loss: DiceCELoss | None = None

    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean, std):
        """
        In-place per-channel normalize for ND volume.
        x: [B, C, ...]
        """
        for c, (m, s) in enumerate(zip(mean, std)):
            x[:, c].sub_(float(m)).div_(float(s))
        return x

    def _get_seg_loss(self, trainer) -> DiceCELoss:

        if self._seg_loss is not None:
            return self._seg_loss

        cfg = trainer.config
        crit_cfg = get_config(cfg, "training.criterion", DictConfig({}))
        include_background = bool(get_config(crit_cfg, "include_background", False))
        squared_pred = bool(get_config(crit_cfg, "squared_pred", False))
        jaccard = bool(get_config(crit_cfg, "jaccard", False))
        lambda_dice = float(get_config(crit_cfg, "lambda_dice", 1.0))
        lambda_ce = float(get_config(crit_cfg, "lambda_ce", 1.0))

        self._seg_loss = DiceCELoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
            squared_pred=squared_pred,
            jaccard=jaccard,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            reduction="mean",
        )
        return self._seg_loss

    # ---------------- Surrogate-step：Update surrogate ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        # data
        x = batch["image"].to(device).float()          # [B,C,...]
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long
        )
        keys: Iterable[int] = batch["key"]

        B, C_in = x.shape[:2]

        # normalization config
        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        # noise: sample-wise
        delta = nb.batch_noise(list(keys)).to(device).float()  # [B,C_in,...]
        if delta.shape[:2] != x.shape[:2]:
            raise RuntimeError(
                f"[UE] noise shape mismatch: noise {tuple(delta.shape)} vs input {tuple(x.shape)}"
            )

        # select surrogate and optimizer
        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        name, s_model = next(iter(trainer.surrogates.items()))
        opt = trainer.opt_surrogates.get(name, None)
        if opt is None:
            raise RuntimeError(f"[UE] No optimizer for surrogate '{name}'.")

        seg_loss_fn = self._get_seg_loss(trainer)

        s_model.train()
        for p in s_model.parameters():
            p.requires_grad = True

        # forward with noisy input
        noisy = (x + delta).clamp(0.0, 1.0)
        xn = noisy.clone()
        self._norm_inplace(xn, mean, std)

        out = s_model(xn)
        logits = out[0] if isinstance(out, (tuple, list)) else out  # [B,C_seg,...]

        loss = seg_loss_fn(logits, y.unsqueeze(1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_val = float(loss.detach().cpu())
        return {
            "surrogate_loss": loss_val,
            "loss": loss_val,
        }

    # ---------------- N-step：Update noise (PGD, sample-wise) ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        # -------- data & config --------
        x = batch["image"].to(device).float()  # [N, C_in, ...]
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long
        )
        keys = batch["key"]
        keys_list: List[int] = list(keys)

        N, C_in = x.shape[:2]

        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        eps = float(get_config(params, "epsilon", 8 / 255.0))
        step_size = float(get_config(params, "step_size", 2 / 255.0))
        num_steps = int(get_config(params, "noise_step", 10))

        # normalization config
        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        seg_loss_fn = self._get_seg_loss(trainer)

        # -------- freeze surrogate --------
        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        _, s_model = next(iter(trainer.surrogates.items()))
        s_model.eval()
        for p in s_model.parameters():
            p.requires_grad = False

        # -------- init / clamp noise --------
        delta_tbl = nb.batch_noise(keys_list).to(device).float()  # [N, C_in, ...]
        if delta_tbl.shape[:2] != x.shape[:2]:
            raise RuntimeError(
                f"[UE] noise shape mismatch: noise {tuple(delta_tbl.shape)} vs input {tuple(x.shape)}"
            )

        delta_tbl = delta_tbl.clamp(-eps, eps)

        last_loss = torch.tensor(0.0, device=device)

        with torch.enable_grad():
            for _ in range(max(1, num_steps)):
                perturb_img = (x + delta_tbl).clamp(0.0, 1.0).detach().requires_grad_(True)
                xn = perturb_img.clone()
                self._norm_inplace(xn, mean, std)

                out = s_model(xn)
                logits = out[0] if isinstance(out, (tuple, list)) else out

                loss = seg_loss_fn(logits, y.unsqueeze(1))
                last_loss = loss.detach()

                (g,) = torch.autograd.grad(
                    loss, perturb_img, retain_graph=False, create_graph=False
                )

                delta_tbl = delta_tbl - step_size * g.sign()

                delta_tbl = delta_tbl.clamp(-eps, eps)

        nb.commit_batch(keys_list, delta_tbl.detach().cpu())

        delta_linf = float(delta_tbl.detach().abs().max().cpu())
        return {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
        }
