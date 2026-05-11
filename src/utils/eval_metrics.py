# file: src/utils/eval_metrics.py
"""
Evaluation metrics for image quality assessment.

Provides:
  - compute_psnr: Peak Signal-to-Noise Ratio
  - compute_ssim: Structural Similarity Index (uses local ssim.py)
  - compute_noise_jacobian_metrics: Noise smoothness metrics
  - IQAPyTorchMetrics: IQA-PyTorch wrapper (if available)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Union
import math

import torch
from torch import Tensor

# Import SSIM from local module
from .ssim import ssim as _ssim_fn


# Check if pyiqa is available
try:
    import pyiqa
    HAS_PYIQA = True
except ImportError:
    HAS_PYIQA = False


def compute_psnr(
    original: Tensor,
    perturbed: Tensor,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        original: Original image tensor [B, C, ...]
        perturbed: Perturbed image tensor [B, C, ...]
        data_range: Value range of images (1.0 for [0,1], 255 for [0,255])
        eps: Small value to avoid log(0)

    Returns:
        PSNR values per sample [B]
    """
    if original.shape != perturbed.shape:
        raise ValueError(
            f"Input shapes must match, got {original.shape} and {perturbed.shape}"
        )

    # Flatten spatial dimensions: [B, C, ...] -> [B, -1]
    batch_size = original.shape[0]
    orig_flat = original.view(batch_size, -1)
    pert_flat = perturbed.view(batch_size, -1)

    # MSE per sample
    mse = torch.mean((orig_flat - pert_flat) ** 2, dim=1)

    # Avoid log(0)
    mse = torch.clamp(mse, min=eps)

    # PSNR = 10 * log10(data_range^2 / MSE)
    psnr = 10.0 * torch.log10((data_range ** 2) / mse)

    return psnr


def compute_ssim(
    original: Tensor,
    perturbed: Tensor,
    data_range: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    size_average: bool = True,
) -> Tensor:
    """
    Compute Structural Similarity Index (SSIM).

    Supports 2D [B,C,H,W] and 3D [B,C,D,H,W] tensors.

    Args:
        original: Original image tensor
        perturbed: Perturbed image tensor
        data_range: Value range of images (1.0 for [0,1], 255 for [0,255])
        win_size: Gaussian window size
        win_sigma: Gaussian window sigma
        size_average: Whether to average over all samples

    Returns:
        SSIM value(s)
    """
    return _ssim_fn(
        original,
        perturbed,
        data_range=data_range,
        win_size=win_size,
        win_sigma=win_sigma,
        size_average=size_average,
    )


def compute_noise_jacobian_metrics(noise: Tensor) -> Dict[str, float]:
    """
    Compute noise smoothness/sparsity metrics via finite differences.

    Metrics:
      - noise_l2: L2 norm (RMS)
      - noise_linf: L-infinity norm
      - noise_tv: Total variation (spatial smoothness)
      - noise_grad_mean: Mean gradient magnitude

    Args:
        noise: Noise tensor [B, C, D, H, W] or [B, C, H, W]

    Returns:
        Dict of metric values
    """
    results = {}

    # Basic norms
    results['noise_l2'] = float(noise.pow(2).mean().sqrt().item())
    results['noise_linf'] = float(noise.abs().max().item())

    # Total variation (sum of absolute differences along spatial dims)
    # Works for both 2D and 3D
    ndim = noise.dim()

    tv = 0.0
    spatial_dims = list(range(2, ndim))  # Skip batch and channel dims

    for dim in spatial_dims:
        # Compute absolute differences along this dimension
        diff = torch.abs(noise.narrow(dim, 1, noise.size(dim) - 1) -
                        noise.narrow(dim, 0, noise.size(dim) - 1))
        tv += diff.sum().item()

    # Normalize by number of elements
    numel = noise.numel()
    results['noise_tv'] = tv / max(numel, 1)

    # Mean gradient magnitude (average of all spatial gradients)
    grad_sum = 0.0
    grad_count = 0

    for dim in spatial_dims:
        diff = noise.narrow(dim, 1, noise.size(dim) - 1) - \
               noise.narrow(dim, 0, noise.size(dim) - 1)
        grad_sum += diff.pow(2).sum().item()
        grad_count += diff.numel()

    if grad_count > 0:
        results['noise_grad_mean'] = math.sqrt(grad_sum / grad_count)
    else:
        results['noise_grad_mean'] = 0.0

    return results


class IQAPyTorchMetrics:
    """
    Wrapper for IQA-PyTorch metrics.

    Provides unified interface for computing PSNR, SSIM, LPIPS, etc.
    using the pyiqa library.
    """

    def __init__(
        self,
        metrics: List[str] = ['psnr', 'ssim'],
        device: Optional[str] = None,
    ):
        """
        Initialize IQA-PyTorch metrics.

        Args:
            metrics: List of metric names (e.g., ['psnr', 'ssim', 'lpips'])
            device: Device to use (auto-detect if None)
        """
        if not HAS_PYIQA:
            raise ImportError(
                "pyiqa is not installed. Install with: pip install pyiqa"
            )

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.metric_names = metrics
        self._metrics = {}

        for name in metrics:
            try:
                self._metrics[name] = pyiqa.create_metric(
                    name, device=self.device
                )
            except Exception as e:
                print(f"Warning: Failed to create metric '{name}': {e}")

    def _ensure_2d_input(self, x: Tensor) -> Tensor:
        """Ensure input is 4D [B,C,H,W] for pyiqa."""
        if x.dim() == 3:
            x = x.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
        elif x.dim() == 5:
            # [B,C,D,H,W] - not supported directly, caller should slice
            raise ValueError(
                "5D input not supported. Use compute_3d_slicewise instead."
            )
        return x

    def _to_rgb_range(self, x: Tensor) -> Tensor:
        """
        Convert to RGB-like format if needed.

        pyiqa typically expects [0,1] range with 3 channels for
        perceptual metrics like LPIPS.
        """
        # Ensure at least 3 channels by repeating if single channel
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            # Take first 3 channels
            x = x[:, :3]
        return x

    def compute_2d(
        self,
        original: Tensor,
        perturbed: Tensor,
    ) -> Dict[str, float]:
        """
        Compute metrics for 2D images.

        Args:
            original: [B, C, H, W] original images
            perturbed: [B, C, H, W] perturbed images

        Returns:
            Dict of metric values
        """
        original = self._ensure_2d_input(original).to(self.device)
        perturbed = self._ensure_2d_input(perturbed).to(self.device)

        results = {}

        for name, metric in self._metrics.items():
            try:
                # Some metrics need RGB
                if name in ['lpips', 'lpips-vgg', 'dists']:
                    orig = self._to_rgb_range(original)
                    pert = self._to_rgb_range(perturbed)
                else:
                    orig = original
                    pert = perturbed

                score = metric(orig, pert)
                results[name] = float(score.mean().item())
            except Exception as e:
                print(f"Warning: Failed to compute {name}: {e}")
                results[name] = float('nan')

        return results

    def compute_3d_slicewise(
        self,
        original: Tensor,
        perturbed: Tensor,
        sample_slices: Optional[int] = None,
        slice_dim: int = 2,
    ) -> Dict[str, float]:
        """
        Compute metrics for 3D volumes by averaging over slices.

        Args:
            original: [B, C, D, H, W] original volumes
            perturbed: [B, C, D, H, W] perturbed volumes
            sample_slices: If set, randomly sample this many slices
            slice_dim: Dimension to slice along (default: 2 for depth)

        Returns:
            Dict of averaged metric values
        """
        if original.dim() != 5:
            raise ValueError(f"Expected 5D input, got {original.dim()}D")

        B, C, D, H, W = original.shape
        num_slices = D

        # Determine which slices to use
        if sample_slices is not None and sample_slices < num_slices:
            indices = torch.randperm(num_slices)[:sample_slices].sort().values
        else:
            indices = torch.arange(num_slices)

        # Accumulate metrics
        metric_sums: Dict[str, float] = {name: 0.0 for name in self._metrics}
        valid_counts: Dict[str, int] = {name: 0 for name in self._metrics}

        for idx in indices:
            # Extract slice: [B, C, H, W]
            orig_slice = original[:, :, idx, :, :]
            pert_slice = perturbed[:, :, idx, :, :]

            slice_metrics = self.compute_2d(orig_slice, pert_slice)

            for name, value in slice_metrics.items():
                if not math.isnan(value):
                    metric_sums[name] += value
                    valid_counts[name] += 1

        # Average
        results = {}
        for name in self._metrics:
            if valid_counts[name] > 0:
                results[name] = metric_sums[name] / valid_counts[name]
            else:
                results[name] = float('nan')

        return results