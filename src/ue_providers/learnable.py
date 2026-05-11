# file: src/ue_providers/learnable.py
from __future__ import annotations
from typing import Any, Dict, Hashable, Iterable, List, Tuple

import numpy as np
import torch

from ..registry import register_provider


def _canon_key(k: Any) -> Hashable:
    '''
    Canonicalize key to a hashable type (for dict index).
    '''
    if torch.is_tensor(k):
        if k.ndim == 0:
            return k.item()
        return tuple(np.asarray(k.cpu()).reshape(-1).tolist())
    if isinstance(k, (np.integer,)):
        return int(k.item())
    if isinstance(k, (np.floating,)):
        return float(k.item())
    return k


def _make_key_index(keys: Iterable[Hashable]) -> Tuple[Dict[Hashable, int], List[Hashable]]:
    canon_keys = [_canon_key(k) for k in keys]
    uniq_list: List[Hashable] = list(dict.fromkeys(canon_keys))
    k2i = {k: i for i, k in enumerate(uniq_list)}
    return k2i, uniq_list


@register_provider("learnable")
class LearnableProvider:
    '''
    EM-style noise cache for 3D segmentation volumes (resident on CPU):

      - Store a noise table Δ[key] ∈ R^{C×D×H×W}.
      - During training:
          batch -> batch_noise(keys)   -> get current Δ on GPU
                -> PGD / gradient step -> updated Δ
                -> commit_batch(keys, updated_Δ) writes back to CPU
    '''

    def __init__(
        self,
        *,
        keys: Iterable[Hashable],
        image_size: Tuple[int, int, int, int],  # (C_in, D, H, W) of model input
        epsilon: float,
    ):
        '''
        Args:
            keys:        Iterable of raw keys (sample identifiers). One entry per unique key.
            image_size:  (C_in, D, H, W) shape of the model input (3D volume).
            epsilon:     L_inf bound in input space. Noise will be clipped to [-eps, eps].
        '''
        self.key2idx, self.uniq_keys = _make_key_index(keys)
        C_in, D, H, W = [int(v) for v in image_size]
        if C_in <= 0 or D <= 0 or H <= 0 or W <= 0:
            raise ValueError(f"[LearnableProvider] Bad image_size={image_size}")
        self.eps: float = float(epsilon)

        N = len(self.uniq_keys)
        if N <= 0:
            raise ValueError("[LearnableProvider] 'keys' must be non-empty.")

        # Whole-volume noise table on CPU
        self._table = torch.empty((N, C_in, D, H, W), dtype=torch.float32, device="cpu")
        self._table.uniform_(-self.eps, +self.eps)

    # ------------------- training-time APIs ------------------- #
    @torch.no_grad()
    def batch_noise(self, keys_raw: Iterable[Hashable]) -> torch.Tensor:
        '''
        Fetch a batch of noise volumes corresponding to `keys_raw`.

        Returns:
            Tensor of shape [B, C, D, H, W] on CPU (detached clone).
        '''
        idx = torch.as_tensor([self.key2idx[_canon_key(k)] for k in keys_raw], dtype=torch.long)
        return self._table.index_select(0, idx).detach().clone()

    @torch.no_grad()
    def commit_batch(self, keys_raw: Iterable[Hashable], updated_noise_cpu: torch.Tensor) -> None:
        '''
        Write back an updated batch of noise volumes.

        Args:
            keys_raw:          Iterable of keys (same order as used for batch_noise).
            updated_noise_cpu: Tensor [B, C, D, H, W] on CPU or GPU.
        '''
        idx = torch.as_tensor([self.key2idx[_canon_key(k)] for k in keys_raw], dtype=torch.long)
        d = updated_noise_cpu
        if d.device.type != "cpu":
            d = d.to("cpu")
        d = d.to(torch.float32)

        if d.ndim != 5 \
           or d.size(0) != idx.size(0) \
           or d.size(1) != self._table.size(1) \
           or d.size(2) != self._table.size(2) \
           or d.size(3) != self._table.size(3) \
           or d.size(4) != self._table.size(4):
            raise RuntimeError(
                f"[UE] commit shape mismatch: got {tuple(d.shape)}, "
                f"expect ({idx.size(0)},"
                f"{self._table.size(1)},"
                f"{self._table.size(2)},"
                f"{self._table.size(3)},"
                f"{self._table.size(4)})"
            )

        d.clamp_(-self.eps, +self.eps)
        self._table.index_copy_(0, idx, d)

    # ------------------- export-time API ------------------- #
    @torch.no_grad()
    def get_noise(self, key_raw: Hashable, perturb_type: str) -> torch.Tensor:
        '''
        Returns:
            noise volume [C, D, H, W], float32, on CPU, clipped to [-eps, eps].
        '''
        i = self.key2idx[_canon_key(key_raw)]
        return self._table[i].clone().clamp_(-self.eps, +self.eps)

    @property
    def channel_count(self) -> int:
        return int(self._table.size(1))
