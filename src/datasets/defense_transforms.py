# file: src/datasets/defense_transforms.py
"""
Defense data augmentation transforms for poison ablation experiments.
All transforms follow the interface: (image, label) -> (image, label)
where image: [C,D,H,W] float32, label: [D,H,W] long.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, Any, Sequence, Tuple

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  1. Gaussian Blur
# ------------------------------------------------------------------ #

def _gaussian_kernel_1d(sigma: float, kernel_size: int) -> torch.Tensor:
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


class GaussianBlur3D:
    """3D separable Gaussian blur via three 1-D conv3d passes."""

    def __init__(self, sigma: float = 1.0, kernel_size: int = 0):
        self.sigma = sigma
        # auto kernel_size: 6*sigma+1, ensure odd
        if kernel_size <= 0:
            kernel_size = int(math.ceil(sigma * 6)) | 1
        self.kernel_size = kernel_size
        self._kernel_1d = _gaussian_kernel_1d(sigma, kernel_size)

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # image: [C,D,H,W]
        k = self._kernel_1d.to(image.device, image.dtype)
        ks = self.kernel_size
        pad = ks // 2
        C = image.shape[0]

        # reshape kernel for depthwise conv along each axis
        # axis D
        kd = k.view(1, 1, ks, 1, 1).expand(C, -1, -1, -1, -1)
        # axis H
        kh = k.view(1, 1, 1, ks, 1).expand(C, -1, -1, -1, -1)
        # axis W
        kw = k.view(1, 1, 1, 1, ks).expand(C, -1, -1, -1, -1)

        x = image.unsqueeze(0)  # [1,C,D,H,W]
        x = F.conv3d(x, kd, padding=(pad, 0, 0), groups=C)
        x = F.conv3d(x, kh, padding=(0, pad, 0), groups=C)
        x = F.conv3d(x, kw, padding=(0, 0, pad), groups=C)
        return x.squeeze(0), label


# ------------------------------------------------------------------ #
#  2. Gamma Correction
# ------------------------------------------------------------------ #

class GammaCorrection:
    """Random gamma correction: x^gamma, gamma sampled from [lo, hi]."""

    def __init__(self, gamma_range: Sequence[float] = (0.7, 1.5)):
        self.lo, self.hi = float(gamma_range[0]), float(gamma_range[1])

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = random.uniform(self.lo, self.hi)
        # image assumed in [0, 1]
        image = image.clamp(min=0.0).pow(gamma)
        return image, label


# ------------------------------------------------------------------ #
#  3. Simulated Low Resolution
# ------------------------------------------------------------------ #

class SimulateLowResolution:
    """
    Downsample then upsample to simulate low-resolution acquisition.
    Only applied along spatial dims (D, H, W).
    """

    def __init__(self, scale: float = 0.5):
        self.scale = float(scale)

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # image: [C,D,H,W]
        orig_shape = image.shape[1:]  # (D,H,W)
        x = image.unsqueeze(0)  # [1,C,D,H,W]
        x = F.interpolate(x, scale_factor=self.scale, mode="trilinear", align_corners=False)
        x = F.interpolate(x, size=orig_shape, mode="trilinear", align_corners=False)
        return x.squeeze(0), label


# ------------------------------------------------------------------ #
#  4. Random Affine (rotation + scale)
# ------------------------------------------------------------------ #

class RandomAffine3D:
    """
    Random 3D affine transform (rotation around each axis + isotropic scale).
    Uses grid_sample; applies to both image and label.
    """

    def __init__(
        self,
        rotation_range: float = 15.0,     # degrees, per axis
        scale_range: Sequence[float] = (0.85, 1.25),
        prob: float = 0.5,
    ):
        self.rotation_range = float(rotation_range)
        self.scale_lo, self.scale_hi = float(scale_range[0]), float(scale_range[1])
        self.prob = float(prob)

    @staticmethod
    def _rot_matrix(ax: float, ay: float, az: float, s: float) -> torch.Tensor:
        """Build 3x4 affine matrix from Euler angles (radians) + scale."""
        cx, sx = math.cos(ax), math.sin(ax)
        cy, sy = math.cos(ay), math.sin(ay)
        cz, sz = math.cos(az), math.sin(az)

        Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
        Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
        Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)

        R = Rz @ Ry @ Rx * s  # [3,3]
        # append zero translation -> [3,4]
        return torch.cat([R, torch.zeros(3, 1)], dim=1)

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.prob:
            return image, label

        deg2rad = math.pi / 180.0
        ax = random.uniform(-self.rotation_range, self.rotation_range) * deg2rad
        ay = random.uniform(-self.rotation_range, self.rotation_range) * deg2rad
        az = random.uniform(-self.rotation_range, self.rotation_range) * deg2rad
        s = random.uniform(self.scale_lo, self.scale_hi)

        theta = self._rot_matrix(ax, ay, az, s).unsqueeze(0)  # [1,3,4]
        theta = theta.to(image.device, image.dtype)

        # image: [C,D,H,W] -> [1,C,D,H,W]
        size = image.shape[1:]  # (D,H,W)
        grid = F.affine_grid(theta, [1, image.shape[0], *size], align_corners=False)
        img_out = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear",
                                padding_mode="border", align_corners=False).squeeze(0)

        # label: [D,H,W] -> [1,1,D,H,W], nearest interpolation
        lbl = label.unsqueeze(0).unsqueeze(0).float()
        lbl_out = F.grid_sample(lbl, grid, mode="nearest",
                                padding_mode="border", align_corners=False)
        lbl_out = lbl_out.squeeze(0).squeeze(0).long()

        return img_out, lbl_out


# ------------------------------------------------------------------ #
#  Factory
# ------------------------------------------------------------------ #

_REGISTRY: Dict[str, type] = {
    "gaussian_blur": GaussianBlur3D,
    "gamma": GammaCorrection,
    "low_resolution": SimulateLowResolution,
    "random_affine": RandomAffine3D,
}


# Each class accepts only these params (used to filter YAML union keys)
_ACCEPTED_PARAMS: Dict[str, set] = {
    "gaussian_blur": {"sigma", "kernel_size"},
    "gamma": {"gamma_range"},
    "low_resolution": {"scale"},
    "random_affine": {"rotation_range", "scale_range", "prob"},
}


def build_defense_transform(cfg: Dict[str, Any]) -> Callable | None:
    """
    Build a defense transform from config dict.

    Because the Hydra YAML declares a union of ALL possible params (with
    defaults), we filter to only the params accepted by the chosen class.

    Returns None if cfg is None, empty, or type is None/"".
    """
    if cfg is None:
        return None

    cfg = dict(cfg)
    dtype = cfg.pop("type", None)
    if dtype is None or str(dtype).strip() == "" or str(dtype).lower() == "null":
        return None
    dtype = str(dtype).lower()

    cls = _REGISTRY.get(dtype)
    if cls is None:
        raise ValueError(
            f"Unknown defense transform type: {dtype!r}. "
            f"Available: {list(_REGISTRY.keys())}"
        )

    accepted = _ACCEPTED_PARAMS.get(dtype, set())
    filtered = {k: v for k, v in cfg.items() if k in accepted and v is not None}
    return cls(**filtered)