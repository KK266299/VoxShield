# file: src/core/ue_artifacts.py
from __future__ import annotations
import os
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.serialization import add_safe_globals

# Make numpy scalar types safe for torch.load (weights_only=False in PyTorch 2.6+)
add_safe_globals([np.core.multiarray.scalar])


# -------------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _quantize_int8(x: torch.Tensor, eps: float) -> Tuple[torch.Tensor, float]:
    """
    Quantize float noise in [-eps, eps] to int8 with a single global scale.

    Args:
        x:  float tensor (any shape), assumed already clipped into [-eps, eps]
        eps: L_inf bound in the same scale as x.

    Returns:
        q:     int8 tensor, same shape as x
        scale: float, s = eps / 127.0  (dequant: x_hat = q * s)
    """
    x = x.to(torch.float32)
    s = float(eps) / 127.0
    if s <= 0:
        raise ValueError(f"epsilon must be > 0, got {eps}")
    q = torch.round(x / s).clamp_(-127, 127).to(torch.int8)
    return q, s


def _json_safe_key(k: Any) -> Any:
    """
    Make raw keys JSON-serializable for manifest:
    - ints/floats/str/bool/None: passthrough
    - list/tuple: convert to list
    - others: fallback to str(k)
    """
    if isinstance(k, (int, float, str, bool)) or k is None:
        return k
    if isinstance(k, (list, tuple)):
        return list(k)
    return str(k)


def _check_same_shape(entries: Sequence[Tuple[Any, torch.Tensor]]) -> Tuple[int, ...]:
    """
    Ensure all tensors in `entries` share the same shape.

    Returns:
        The common tensor shape as a tuple (e.g., (C,D,H,W)).
    """
    if not entries:
        raise ValueError("entries is empty in _check_same_shape")
    first_shape = tuple(entries[0][1].shape)
    for i, (_, t) in enumerate(entries):
        if tuple(t.shape) != first_shape:
            raise ValueError(
                f"All noise tensors must share the same shape. "
                f"Entry {i} has {tuple(t.shape)} vs first {first_shape}."
            )
    return first_shape


# -------------------------------------------------------------------------
# List-based writers (generic; used for small/medium-scale exports)
# -------------------------------------------------------------------------

def write_shards(
    store_dir: str,
    entries: Sequence[Tuple[Any, torch.Tensor]],  # (raw_key, tensor)
    eps: float,
    shard_size: int,
    perturb_type: str,                         # "samplewise" | "classwise"
    key_spec: Optional[Dict[str, Any]] = None, # describe how key is derived
) -> str:
    """
    Write perturbations in shard format with raw keys (generic N-D).

    Manifest schema (NEW):
      {
        "version": 1,
        "strategy": "shards",
        "dtype": "int8",
        "scale": <float s>,          # dequant: x_hat = q * s
        "image_size": [C,D,H,W],     # or [C,H,W] for 2D, etc.
        "perturb_type": "samplewise" | "classwise",
        "key_spec": {...},           # source/canonicalization/namespace
        "entries": [
          {"path": "shards/shard_00000.pt", "keys": [<raw_key0>, <raw_key1>, ...]},
          ...
        ]
      }
    """
    if not entries:
        raise ValueError("No entries to write for shards")

    store_dir = _ensure_dir(store_dir)
    shards_dir = _ensure_dir(os.path.join(store_dir, "shards"))
    shape = _check_same_shape(entries)  # e.g., (C,D,H,W) or (C,H,W)

    # single global scale; compute directly
    scale = float(eps) / 127.0

    manifest: Dict[str, Any] = {
        "version": 1,
        "strategy": "shards",
        "dtype": "int8",
        "scale": scale,
        "image_size": list(shape),
        "perturb_type": str(perturb_type),
        "key_spec": dict(key_spec or {}),
        "entries": [],
    }

    n = len(entries)
    n_shards = math.ceil(n / shard_size)

    for si in range(n_shards):
        seg = entries[si * shard_size : min((si + 1) * shard_size, n)]
        raw_keys = [k for (k, _) in seg]  # keep raw keys in .pt
        raw_keys_json = [_json_safe_key(k) for k in raw_keys]

        # stack & quantize
        stack = torch.stack(
            [t.to(torch.float32) for (_, t) in seg],
            dim=0
        )  # [M,...]
        stack = torch.clamp(stack, -eps, eps)
        q, _ = _quantize_int8(stack, eps=eps)  # quantize with the same eps -> same scale

        shard_path = os.path.join(shards_dir, f"shard_{si:05d}.pt")
        torch.save({"keys": raw_keys, "data": q}, shard_path)
        manifest["entries"].append({
            "path": os.path.relpath(shard_path, store_dir),
            "keys": raw_keys_json,
        })

    manifest_path = os.path.join(store_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def write_files(
    store_dir: str,
    entries: Sequence[Tuple[Any, torch.Tensor]],  # (raw_key, tensor)
    eps: float,
    perturb_type: str,                         # "samplewise" | "classwise"
    key_spec: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Write perturbations as one file per key (raw keys are used in manifest).

    Manifest schema (NEW):
      {
        "version": 1,
        "strategy": "files",
        "dtype": "int8",
        "scale": <float s>,          # dequant: x_hat = q * s
        "image_size": [...],
        "perturb_type": "samplewise" | "classwise",
        "key_spec": {...},
        "entries": [
          {"path": "files/<index>.pt", "keys": [<raw_key>]},
          ...
        ]
      }
    """
    if not entries:
        raise ValueError("No entries to write for files")

    store_dir = _ensure_dir(store_dir)
    files_dir = _ensure_dir(os.path.join(store_dir, "files"))
    shape = _check_same_shape(entries)

    scale = float(eps) / 127.0

    manifest_entries: List[Dict[str, Any]] = []
    for i, (raw_key, noise) in enumerate(entries):
        noise_f = torch.clamp(noise.to(torch.float32), -eps, eps)
        q, _ = _quantize_int8(noise_f, eps=eps)
        filename = f"{i:08d}.pt"
        torch.save({"keys": [raw_key], "data": q}, os.path.join(files_dir, filename))
        manifest_entries.append({
            "path": f"files/{filename}",
            "keys": [_json_safe_key(raw_key)],
        })

    manifest = {
        "version": 1,
        "strategy": "files",
        "dtype": "int8",
        "scale": scale,
        "image_size": list(shape),
        "perturb_type": str(perturb_type),
        "key_spec": dict(key_spec or {}),
        "entries": manifest_entries,
    }

    manifest_path = os.path.join(store_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


# -------------------------------------------------------------------------
# Streaming shards writer (backward compatibility for maybe_generate_ue_artifacts)
# -------------------------------------------------------------------------

def save_shards(
    store_dir: str,
    tensor_iter: Iterable[torch.Tensor],
    keys_iter: Iterable[Any],
    image_size: Tuple[int, ...],
    perturb_type: str,
    shard_size: int,
    dtype: str,
    scale: float,
    mapping: Optional[Dict[str, Any]] = None,
    manifest_extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Streaming variant used by legacy code (e.g., maybe_generate_ue_artifacts).

    Args:
        store_dir: directory to store shards and manifest.
        tensor_iter: iterable over noise tensors, each already clipped to [-eps, eps].
        keys_iter: iterable over raw keys, same order as tensor_iter.
        image_size: expected tensor shape (excluding batch dim), e.g. (C,H,W) or (C,D,H,W).
        perturb_type: "samplewise" or "classwise".
        shard_size: max number of entries per shard file.
        dtype: only "int8" is currently supported.
        scale: dequant scale s (x_hat = q * s); typically eps/127.
        mapping: optional dict to store in manifest["mapping"].
        manifest_extra: optional dict to merge into top-level manifest.

    Returns:
        Path to manifest.json.
    """
    if dtype.lower() != "int8":
        raise ValueError(f"save_shards currently only supports dtype='int8', got {dtype!r}")

    store_dir = _ensure_dir(store_dir)
    shards_dir = _ensure_dir(os.path.join(store_dir, "shards"))

    manifest: Dict[str, Any] = {
        "version": 1,
        "strategy": "shards",
        "dtype": "int8",
        "scale": float(scale),
        "image_size": list(image_size),
        "perturb_type": str(perturb_type),
        "key_spec": {},  # older path didn't specify; caller can extend if needed
        "entries": [],
    }
    if mapping is not None:
        manifest["mapping"] = mapping
    if manifest_extra is not None:
        manifest.update(dict(manifest_extra))

    # streaming accumulation
    buf_keys: List[Any] = []
    buf_tensors: List[torch.Tensor] = []

    def _flush(shard_index: int, keys_batch: List[Any], tensors_batch: List[torch.Tensor]) -> None:
        if not keys_batch:
            return
        stack = torch.stack([t.to(torch.float32) for t in tensors_batch], dim=0)
        q = torch.round(stack / scale).clamp_(-127, 127).to(torch.int8)
        shard_filename = f"shard_{shard_index:05d}.pt"
        shard_path = os.path.join(shards_dir, shard_filename)
        torch.save({"keys": keys_batch, "data": q}, shard_path)
        manifest["entries"].append({
            "path": os.path.relpath(shard_path, store_dir),
            "keys": [_json_safe_key(k) for k in keys_batch],
        })

    shard_idx = 0
    for raw_key, tensor in zip(keys_iter, tensor_iter):
        buf_keys.append(raw_key)
        buf_tensors.append(tensor)
        if len(buf_keys) >= shard_size:
            _flush(shard_idx, buf_keys, buf_tensors)
            shard_idx += 1
            buf_keys = []
            buf_tensors = []

    # flush tail
    if buf_keys:
        _flush(shard_idx, buf_keys, buf_tensors)

    manifest_path = os.path.join(store_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


# -------------------------------------------------------------------------
# Accessor
# -------------------------------------------------------------------------

class UEShardsAccessor:
    """
    Accessor for UE artifacts written by `write_shards`, `write_files`, or `save_shards`.

    Manifest schema (NEW):
      {
        "version": 1,
        "strategy": "shards" | "files",
        "dtype": "int8",
        "scale": <float s>,            # dequant: x_hat = q * s
        "image_size": [...],
        "perturb_type": "samplewise" | "classwise",
        "key_spec": {...},             # optional, describes key's source/namespace
        "entries": [
          {"path": "<relpath>", "keys": [ ... ]},  # shards: 多 key | files: 单 key
          ...
        ],
        ... other optional metadata ...
      }
    """

    def __init__(self, manifest: Dict[str, Any], root_dir: str):
        self.mf = manifest
        self.root = root_dir

        self.strategy = str(manifest["strategy"]).lower()
        if self.strategy not in ("shards", "files"):
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.dtype = str(manifest["dtype"]).lower()
        if self.dtype != "int8":
            raise ValueError("Only dtype=int8 is supported in this accessor.")

        self.scale: float = float(manifest["scale"])  # dequant: x_hat = q * scale
        self.image_size = tuple(manifest.get("image_size", []))

        self.perturb_type = str(manifest.get("perturb_type"))

        self.key_spec = dict(manifest.get("key_spec", {}))
        self.entries: List[Dict[str, Any]] = list(manifest.get("entries", []))

        # Build index: JSON-safe key -> (abs_path, offset or None)
        self._key_to_loc: Dict[Any, Tuple[str, Optional[int]]] = {}
        for en in self.entries:
            rel = en["path"]
            abs_path = os.path.join(self.root, rel)
            keys = [k for k in en["keys"]]

            if self.strategy == "files":
                if len(keys) != 1:
                    raise AssertionError("files strategy expects exactly one key per entry")
                canon_k = _json_safe_key(keys[0])
                self._key_to_loc[canon_k] = (abs_path, None)
            else:
                for i, k in enumerate(keys):
                    canon_k = _json_safe_key(k)
                    self._key_to_loc[canon_k] = (abs_path, i)

        # Tiny cache: path -> loaded object
        self._cache: Dict[str, Any] = {}

    # ---- constructors ----

    @classmethod
    def from_manifest(cls, manifest_path: str, root_dir: Optional[str] = None) -> "UEShardsAccessor":
        with open(manifest_path, "r") as f:
            mf = json.load(f)
        root = root_dir or os.path.dirname(manifest_path)
        return cls(mf, root)

    # ---- public API ----

    def get(self, key_raw: Any, perturb_type: Optional[str] = None) -> torch.Tensor:
        """
        Return a dequantized tensor (same shape as stored, e.g. [C,D,H,W]) in float32.

        Args:
            key_raw: the same raw key as exported.
            perturb_type: optional check; if provided and not consistent with manifest, raise error.
        """
        if perturb_type is not None and self.perturb_type and str(perturb_type) != self.perturb_type:
            raise ValueError(
                f"perturb_type mismatch: accessor has '{self.perturb_type}' vs request '{perturb_type}'"
            )

        ck = _json_safe_key(key_raw)
        if ck not in self._key_to_loc:
            raise KeyError(f"Key {key_raw!r} not found in manifest.")

        path, off = self._key_to_loc[ck]
        if path not in self._cache:
            try:
                # PyTorch 2.6+ defaults weights_only=True; explicitly disable.
                self._cache[path] = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                # Older torch without weights_only
                self._cache[path] = torch.load(path, map_location="cpu")

        blob = self._cache[path]

        if self.strategy == "files":
            q = blob["data"]  # [C,...] int8
            return self._dequant(q)

        # shards
        data = blob["data"]  # [M,C,...] int8
        if off is None:
            raise AssertionError("shards strategy requires an offset")
        q = data[int(off)]
        return self._dequant(q)

    def keys(self) -> List[Any]:
        """Return all available keys (JSON-safe form)."""
        return list(self._key_to_loc.keys())

    # ---- dequant ----

    def _dequant(self, q: torch.Tensor) -> torch.Tensor:
        # x_hat = q * scale
        return q.to(torch.float32) * self.scale
