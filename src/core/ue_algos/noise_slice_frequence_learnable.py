# file: src/core/ue_algos/noise_slice_frequence_learnable.py

from __future__ import annotations
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from monai.losses import DiceCELoss
from monai.networks.nets import UNet as MonaiUNet

from ...registry import register_plugin
from ...utils.config import get_config, require_config
from ...utils.logger import get_logger


# ────────────────────────── Helper modules ────────────────────────── #

def _build_noise_unet(cfg: DictConfig, in_channels: int, spatial_dims: int = 3) -> nn.Module:
    channels = list(get_config(cfg, "channels", [16, 32, 64, 128]))
    strides = list(get_config(cfg, "strides", [2, 2, 2]))
    num_res_units = int(get_config(cfg, "num_res_units", 1))
    act = get_config(cfg, "act", "LEAKYRELU")
    norm = get_config(cfg, "norm", "INSTANCE")
    dropout = float(get_config(cfg, "dropout", 0.0))

    return MonaiUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=in_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        act=act,
        norm=norm,
        dropout=dropout,
    )


class NoiseUNetWrapper(nn.Module):
    def __init__(self, unet: nn.Module, epsilon: float = 8 / 255):
        super().__init__()
        self.unet = unet
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.unet(x)) * self.epsilon


class SoftROIMask(nn.Module):
    def __init__(
        self,
        soft_edge: bool = True,
        dilate_iterations: int = 2,
        dilate_kernel_size: int = 3,
        gaussian_sigma: float = 2.0,
    ):
        super().__init__()
        self.soft_edge = soft_edge
        self.dilate_iterations = dilate_iterations
        self.dilate_kernel_size = dilate_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self._gaussian_kernel = None

    def _build_gaussian_kernel(self, device, dtype):
        sigma = self.gaussian_sigma
        ks = int(6 * sigma + 1)
        if ks % 2 == 0:
            ks += 1
        coords = torch.arange(ks, device=device, dtype=dtype) - ks // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        k3d = g.view(-1, 1, 1) * g.view(1, -1, 1) * g.view(1, 1, -1)
        k3d = k3d / k3d.sum()
        return k3d.view(1, 1, ks, ks, ks)

    def forward(self, label: torch.Tensor, num_channels: int) -> torch.Tensor:
        if label.dim() == 5:
            label = label.squeeze(1)
        mask = (label > 0).float().unsqueeze(1)

        if self.soft_edge:
            if self.dilate_iterations > 0 and self.dilate_kernel_size > 0:
                k = self.dilate_kernel_size
                p = k // 2
                for _ in range(self.dilate_iterations):
                    mask = F.max_pool3d(mask, kernel_size=k, stride=1, padding=p)
            if self.gaussian_sigma > 0:
                if self._gaussian_kernel is None:
                    self._gaussian_kernel = self._build_gaussian_kernel(label.device, torch.float32)
                kernel = self._gaussian_kernel.to(device=label.device, dtype=torch.float32)
                p = kernel.shape[-1] // 2
                mask = F.pad(mask, (p, p, p, p, p, p), mode='replicate')
                mask = F.conv3d(mask, kernel)
            mask = mask / mask.max().clamp_min(1e-6)

        return mask.expand(-1, num_channels, -1, -1, -1)


class LogitsDivergenceLoss(nn.Module):
    """
    Compute logits divergence loss between clean and noisy predictions.

    Uses L1 norm of logits difference.
    The loss is negated to maximize divergence (since we minimize loss).
    """

    def __init__(
        self,
        weight: float = 1.0,
    ):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        logits_clean: torch.Tensor,
        logits_noisy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L1 divergence loss.

        Args:
            logits_clean: [B, C, D, H, W] predictions on clean images
            logits_noisy: [B, C, D, H, W] predictions on noisy images

        Returns:
            loss: Scalar tensor (negative divergence for maximization)
        """
        diff = logits_noisy - logits_clean
        divergence = diff.abs().mean()
        return -self.weight * divergence


# ────────────────────────── Main Plugin ────────────────────────── #

@register_plugin("noise_slice_frequence_learnable")
class NoiseSliceFrequenceLearnable:
    """
    Noise UNet with L1 logits divergence loss (no frequency domain mask).

    The noise UNet generates perturbations that are applied to input images.
    An L1 logits divergence loss encourages the perturbations to maximally
    alter the surrogate model's predictions.
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._noise_unet: NoiseUNetWrapper | None = None
        self._opt_unet: torch.optim.Optimizer | None = None
        self._roi_mask_builder: SoftROIMask | None = None
        self._initialized: bool = False
        # Z-axis diversity regularization settings
        self._z_diversity_weight: float = 0.0
        # Logits divergence loss settings
        self._logits_div_loss: LogitsDivergenceLoss | None = None
        self._logits_div_enabled: bool = True
        self.logger = get_logger()

    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean, std):
        for c, (m, s) in enumerate(zip(mean, std)):
            x[:, c].sub_(float(m)).div_(float(s))
        return x

    def _get_seg_loss(self, trainer) -> DiceCELoss:
        if self._seg_loss is not None:
            return self._seg_loss
        cfg = trainer.config
        crit_cfg = get_config(cfg, "training.criterion", DictConfig({}))
        self._seg_loss = DiceCELoss(
            include_background=bool(get_config(crit_cfg, "include_background", False)),
            to_onehot_y=True,
            softmax=True,
            squared_pred=bool(get_config(crit_cfg, "squared_pred", False)),
            jaccard=bool(get_config(crit_cfg, "jaccard", False)),
            lambda_dice=float(get_config(crit_cfg, "lambda_dice", 1.0)),
            lambda_ce=float(get_config(crit_cfg, "lambda_ce", 1.0)),
            reduction="mean",
        )
        return self._seg_loss

    def _init_components(self, trainer, in_channels: int, spatial_dims: int = 3):
        if self._initialized:
            return

        cfg = trainer.config
        device = trainer.device
        params = get_config(cfg, "ue.algorithm.params", DictConfig({}))

        eps = float(get_config(params, "epsilon", 8 / 255))

        # Noise UNet
        noise_unet_cfg = get_config(cfg, "ue.noise_unet", DictConfig({}))
        base_unet = _build_noise_unet(noise_unet_cfg, in_channels, spatial_dims)
        self._noise_unet = NoiseUNetWrapper(base_unet, epsilon=eps).to(device)

        # Optimizer
        opt_cfg = get_config(noise_unet_cfg, "optimizer", DictConfig({}))
        lr = float(get_config(opt_cfg, "lr", 1e-4))
        wd = float(get_config(opt_cfg, "weight_decay", 1e-5))
        betas = tuple(get_config(opt_cfg, "betas", (0.9, 0.999)))

        self._opt_unet = torch.optim.Adam(
            self._noise_unet.parameters(), lr=lr, weight_decay=wd, betas=betas,
        )

        # ROI mask
        self._roi_aware = bool(get_config(params, "roi_aware", True))
        self._roi_mask_builder = SoftROIMask(
            soft_edge=bool(get_config(params, "soft_edge", True)),
            dilate_iterations=int(get_config(params, "dilate_iterations", 2)),
            dilate_kernel_size=int(get_config(params, "dilate_kernel_size", 3)),
            gaussian_sigma=float(get_config(params, "gaussian_sigma", 2.0)),
        )

        # Z-axis diversity regularization (weight=0 disables it)
        self._z_diversity_weight = float(get_config(params, "z_diversity_weight", 0.0))

        # Logits divergence loss (L1)
        self._logits_div_enabled = bool(get_config(params, "logits_div_enabled", True))
        logits_div_weight = float(get_config(params, "logits_div_weight", 0.01))
        if self._logits_div_enabled and logits_div_weight > 0:
            self._logits_div_loss = LogitsDivergenceLoss(
                weight=logits_div_weight,
            )
        else:
            self._logits_div_loss = None
            logits_div_weight = 0.0

        self._initialized = True

        self.logger.info(
            f"[FreqLearnable] Initialized: in_ch={in_channels}, eps={eps:.6f}, "
            f"z_diversity_weight={self._z_diversity_weight:.4f}, "
            f"logits_div_enabled={self._logits_div_enabled}, logits_div_mode=l1, "
            f"logits_div_weight={logits_div_weight:.4f}"
        )

    def on_noise_epoch_end(self, trainer, epoch: int):
        """Called by ue_trainer at end of each noise epoch (no-op without cutoffs)."""
        pass

    # ────────────── S-step: update surrogate ────────────── #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        x = batch["image"].to(device).float()
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long,
        )
        keys: Iterable[int] = batch["key"]
        B, C_in = x.shape[:2]
        self._init_components(trainer, C_in, len(x.shape) - 2)

        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        delta = nb.batch_noise(list(keys)).to(device).float()
        if delta.shape[:2] != x.shape[:2]:
            raise RuntimeError(f"[UE] noise shape mismatch: {tuple(delta.shape)} vs {tuple(x.shape)}")

        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        name, s_model = next(iter(trainer.surrogates.items()))
        opt = trainer.opt_surrogates.get(name)
        if opt is None:
            raise RuntimeError(f"[UE] No optimizer for surrogate '{name}'.")

        seg_loss_fn = self._get_seg_loss(trainer)
        s_model.train()
        for p in s_model.parameters():
            p.requires_grad = True

        noisy = (x + delta).clamp(0.0, 1.0)
        xn = noisy.clone()
        self._norm_inplace(xn, mean, std)

        out = s_model(xn)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        loss = seg_loss_fn(logits, y.unsqueeze(1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        return {"surrogate_loss": float(loss.detach().cpu()), "loss": float(loss.detach().cpu())}

    # ────────────── N-step: update noise ────────────── #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update noise UNet parameters.

        Data flow:
          x → NoiseUNet → δ_raw → (ROI mask) → δ
          δ → surrogate → DiceCE Loss + L1 logits divergence
          Loss.backward() → opt_unet.step()
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        x = batch["image"].to(device).float()
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long,
        )
        keys_list: List[int] = list(batch["key"])

        B, C_in = x.shape[:2]
        self._init_components(trainer, C_in, len(x.shape) - 2)

        params = require_config(require_config(cfg, "ue.algorithm"), "params")
        eps = float(get_config(params, "epsilon", 8 / 255.0))
        num_steps = int(get_config(params, "noise_step", 1))

        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        seg_loss_fn = self._get_seg_loss(trainer)

        # Freeze surrogate
        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        _, s_model = next(iter(trainer.surrogates.items()))
        s_model.eval()
        for p in s_model.parameters():
            p.requires_grad = False

        # ROI mask
        roi_mask = self._roi_mask_builder(y, C_in).to(device) if self._roi_aware else None

        # Get clean image predictions (for logits divergence loss)
        logits_clean = None
        if self._logits_div_enabled and self._logits_div_loss is not None:
            with torch.no_grad():
                x_clean_norm = x.clone()
                self._norm_inplace(x_clean_norm, mean, std)
                out_clean = s_model(x_clean_norm)
                logits_clean = out_clean[0] if isinstance(out_clean, (tuple, list)) else out_clean
                logits_clean = logits_clean.detach()

        self._noise_unet.train()
        last_loss = torch.tensor(0.0, device=device)
        last_z_diversity_loss = torch.tensor(0.0, device=device)
        last_div_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, num_steps)):
            # NoiseUNet forward
            delta_raw = self._noise_unet(x)

            if roi_mask is not None:
                delta = delta_raw * roi_mask
            else:
                delta = delta_raw
            delta = delta.clamp(-eps, eps)

            perturb_img = (x + delta).clamp(0.0, 1.0)
            xn = perturb_img.clone()
            self._norm_inplace(xn, mean, std)

            out = s_model(xn)
            logits_noisy = out[0] if isinstance(out, (tuple, list)) else out
            seg_loss = seg_loss_fn(logits_noisy, y.unsqueeze(1))

            # Start with seg_loss
            loss = seg_loss

            # Z-axis diversity loss (if weight > 0)
            if self._z_diversity_weight > 0:
                z_diversity = self._compute_z_diversity(delta)
                z_diversity_loss = -z_diversity  # negative because we want to maximize
                loss = loss + self._z_diversity_weight * z_diversity_loss
                last_z_diversity_loss = z_diversity_loss.detach()

            # Logits divergence loss (L1)
            if self._logits_div_loss is not None and logits_clean is not None:
                div_loss = self._logits_div_loss(logits_clean, logits_noisy)
                loss = loss + div_loss
                last_div_loss = div_loss.detach()

            last_loss = loss.detach()

            self._opt_unet.zero_grad(set_to_none=True)
            loss.backward()
            self._opt_unet.step()

        # Store final noise to backend
        self._noise_unet.eval()
        with torch.no_grad():
            final_noise = self._noise_unet(x)
            if roi_mask is not None:
                final_delta = final_noise * roi_mask
            else:
                final_delta = final_noise
            final_delta = final_delta.clamp(-eps, eps)

        nb.commit_batch(keys_list, final_delta.detach().cpu())

        delta_linf = float(final_delta.detach().abs().max().cpu())

        with torch.no_grad():
            z_diversity_value = self._compute_z_diversity(final_delta)

            # Compute final logits divergence for logging
            logits_diff_l1 = 0.0
            if logits_clean is not None:
                perturb_final = (x + final_delta).clamp(0.0, 1.0)
                xn_final = perturb_final.clone()
                self._norm_inplace(xn_final, mean, std)
                out_final = s_model(xn_final)
                logits_final = out_final[0] if isinstance(out_final, (tuple, list)) else out_final
                logits_diff_l1 = (logits_final - logits_clean).abs().mean().cpu().item()

        result = {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
            "z_diversity": float(z_diversity_value.cpu()),
        }

        # Add z_diversity_loss to result (if enabled)
        if self._z_diversity_weight > 0:
            result["z_diversity_loss"] = float(last_z_diversity_loss.cpu())

        # Add logits_div metrics to result (if enabled)
        if self._logits_div_loss is not None:
            result["div_loss"] = float(last_div_loss.cpu())
            result["logits_diff_l1"] = logits_diff_l1

        return result

    def _compute_z_diversity(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Compute z-axis inter-slice diversity in frequency domain.

        Computes the mean L2 distance between adjacent slices after 2D FFT,
        encouraging noise to have high variation along the z-axis.

        Args:
            delta: [B, C, D, H, W] noise tensor

        Returns:
            z_diversity: Scalar tensor representing mean inter-slice L2 difference
        """
        # Apply 2D FFT on each slice (xy-plane)
        delta_fft_2d = torch.fft.fft2(delta, dim=(-2, -1))  # [B, C, D, H, W] complex

        # Compute magnitude spectrum for each slice
        delta_fft_mag = delta_fft_2d.abs()  # [B, C, D, H, W]

        # Compute L2 difference between adjacent slices along z-axis
        slice_diff = delta_fft_mag[:, :, 1:, :, :] - delta_fft_mag[:, :, :-1, :, :]  # [B, C, D-1, H, W]

        # Compute L2 norm for each pair of slices
        l2_per_pair = torch.sqrt((slice_diff ** 2).sum(dim=(-2, -1)) + 1e-10)  # [B, C, D-1]

        # Mean over all pairs, channels, and batches
        z_diversity = l2_per_pair.mean()

        return z_diversity
