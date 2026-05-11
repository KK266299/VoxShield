from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, Union
import torch
import numpy as np
import hashlib
from ..registry import register_provider


@register_provider("random_noise")
class RandomNoiseProvider:
    def __init__(
        self,
        *,
        epsilon: float,
        image_size: Union[Tuple[int, int, int], Tuple[int, int, int, int]],
        seed: int = 0,
        mode: str = "uniform",
        params: Optional[Dict[str, Any]] = None,
        tied_channels: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.eps = float(epsilon)

        # Support both 2D (C, H, W) and 3D (C, D, H, W)
        size_tuple = tuple(map(int, image_size))
        if len(size_tuple) == 3:
            self.C, self.H, self.W = size_tuple
            self.D = None
            self.is_3d = False
        elif len(size_tuple) == 4:
            self.C, self.D, self.H, self.W = size_tuple
            self.is_3d = True
        else:
            raise ValueError(
                f"[RandomNoise] image_size must be (C,H,W) or (C,D,H,W), got {size_tuple}"
            )

        self.seed = int(seed)
        self.mode = str(mode).lower()
        self.params = params or {}
        self.tied_channels = bool(tied_channels)
        self.device = device
        self.out_dtype = dtype

        _valid = {"uniform", "gaussian", "rademacher", "saltpepper", "sparse"}
        if self.mode not in _valid:
            raise ValueError(f"[RandomNoise] Unsupported mode: {self.mode}. "
                             f"Choose from {_valid}.")

    # ---------- public API ----------

    @torch.no_grad()
    def get_noise(self, key, key_type: str) -> torch.Tensor:
        if key_type not in ("classwise", "samplewise"):
            raise ValueError(f"[RandomNoise] Unknown key_type: {key_type}")

        kid = self._hashable_int(key)
        if key_type == "classwise":
            lseed = (self.seed * 1315423911 + kid) & 0x7FFFFFFF
        else:
            lseed = (self.seed * 2654435761 + kid) & 0x7FFFFFFF

        if self.is_3d:
            # 3D mode: generate [C, D, H, W] noise
            if self.tied_channels:
                one = self._synthesize_one_3d(lseed)  # [1, D, H, W]
                noise = one.expand(self.C, self.D, self.H, self.W).contiguous()
            else:
                noise_list = []
                for c in range(self.C):
                    lseed_c = (lseed + (c + 1) * 104729) & 0x7FFFFFFF
                    noise_list.append(self._synthesize_one_3d(lseed_c))  # [1, D, H, W]
                noise = torch.cat(noise_list, dim=0)  # [C, D, H, W]
        else:
            # 2D mode: generate [C, H, W] noise
            if self.tied_channels:
                one = self._synthesize_one_2d(lseed)  # [1, H, W]
                noise = one.expand(self.C, self.H, self.W).contiguous()
            else:
                noise_list = []
                for c in range(self.C):
                    lseed_c = (lseed + (c + 1) * 104729) & 0x7FFFFFFF
                    noise_list.append(self._synthesize_one_2d(lseed_c))  # [1, H, W]
                noise = torch.cat(noise_list, dim=0)  # [C, H, W]

        noise = noise.clamp(min=-self.eps, max=self.eps).to(self.out_dtype)
        return noise.cpu()

    # ---------- internals ----------

    @staticmethod
    def _hashable_int(x) -> int:
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, torch.Tensor):
            if x.numel() != 1:
                raise ValueError("Key tensor must be scalar.")
            return int(x.item())
        if isinstance(x, (str, bytes, bytearray)):
            h = hashlib.sha1(str(x).encode("utf-8")).hexdigest()
            return int(h[:16], 16)
        try:
            return int(x)
        except Exception:
            h = hashlib.sha1(repr(x).encode("utf-8")).hexdigest()
            return int(h[:16], 16)

    @torch.no_grad()
    def _synthesize_one_2d(self, local_seed: int) -> torch.Tensor:
        """
        Generate one-channel 2D noise [1, H, W], deterministic with respect to local_seed.
        All modes will finally normalize/clip to [-eps, eps].
        """
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(local_seed))

        d = self.device if self.device is not None else "cpu"
        shape = (1, self.H, self.W)

        return self._apply_noise_mode(gen, d, shape)

    @torch.no_grad()
    def _synthesize_one_3d(self, local_seed: int) -> torch.Tensor:
        """
        Generate one-channel 3D noise [1, D, H, W], deterministic with respect to local_seed.
        All modes will finally normalize/clip to [-eps, eps].
        """
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(local_seed))

        d = self.device if self.device is not None else "cpu"
        shape = (1, self.D, self.H, self.W)

        return self._apply_noise_mode(gen, d, shape)

    @torch.no_grad()
    def _apply_noise_mode(
        self,
        gen: torch.Generator,
        device,
        shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Apply noise generation based on mode. Works for both 2D and 3D shapes.
        """
        if self.mode == "uniform":
            arr = torch.empty(shape, dtype=torch.float32, device="cpu")
            arr.uniform_(-1.0, 1.0, generator=gen)
            arr = arr.to(device)
            # directly scale to [-eps, eps]
            arr = arr * self.eps
            return arr

        elif self.mode == "gaussian":
            arr = torch.randn(shape, generator=gen, dtype=torch.float32, device="cpu")
            arr = arr.to(device)
            # L∞ normalization to avoid Gaussian tail exceeding eps
            max_abs = arr.abs().amax() + 1e-12
            arr = arr / max_abs * self.eps
            return arr

        elif self.mode == "rademacher":
            # with 0.5 probability, set to +eps or -eps
            bern = torch.rand(shape, generator=gen, dtype=torch.float32, device="cpu")
            sign = torch.where(bern < 0.5, -1.0, 1.0)
            arr = (sign * self.eps).to(device)
            return arr

        elif self.mode == "saltpepper":
            # sparse pulse noise: set to ±eps with probability p, others are 0
            p = float(self.params.get("p", 0.01))
            pepper_prob = float(self.params.get("pepper_prob", 0.5))
            r = torch.rand(shape, generator=gen, dtype=torch.float32, device="cpu")
            rp = torch.rand(shape, generator=gen, dtype=torch.float32, device="cpu")

            zeros = torch.zeros(shape, dtype=torch.float32)
            pos = torch.full(shape, self.eps, dtype=torch.float32)
            neg = torch.full(shape, -self.eps, dtype=torch.float32)

            arr = torch.where(r > p, zeros, torch.where(rp < pepper_prob, neg, pos))
            arr = arr.to(device)
            return arr

        elif self.mode == "sparse":
            # sparse additive: sample U(-1,1) with probability q, others are 0, then normalize/clip to eps
            q = float(self.params.get("q", 0.05))
            mask = (torch.rand(shape, generator=gen, dtype=torch.float32, device="cpu") < q).float()
            vals = torch.empty(shape, dtype=torch.float32, device="cpu").uniform_(-1.0, 1.0, generator=gen)
            arr = (mask * vals).to(device) * self.eps
            return arr

        else:
            raise RuntimeError(f"[RandomNoise] Unknown mode at runtime: {self.mode}")