# file: src/core/ue_keys.py
from __future__ import annotations
from typing import Any, Dict, List
from torch.utils.data import ConcatDataset

def _get_by_path(sample: Dict[str, Any], path: str) -> Any:
    cur = sample
    for tok in path.split("."):
        if not isinstance(cur, dict) or tok not in cur:
            raise KeyError(f"[ue_keys] Path '{path}' not found (at '{tok}')")
        cur = cur[tok]
    return cur

def _canon_str(s: str, *, lower: bool, strip: bool) -> str:
    if strip:
        s = s.strip()
    if lower:
        s = s.lower()
    return s

def extract_key(sample: Dict[str, Any], idx: int, key_spec: Dict[str, Any]) -> Any:
    """
    Unified key extraction:
    - from: index | field | filename
    - field supports dot paths; strings are normalized by key_spec.lower/strip
    - returns raw-key (str/int/tuple/etc), never forces casting
    """
    kfrom = str(key_spec.get("from", "field")).lower()
    lower = bool(key_spec.get("lower", True))
    strip = bool(key_spec.get("strip", True))

    if kfrom == "index":
        return idx

    if kfrom == "field":
        field = key_spec.get("field", None)
        if not field:
            raise ValueError("[ue_keys] key_spec.field is required when from='field'")
        val = _get_by_path(sample, field)
        if isinstance(val, str):
            val = _canon_str(val, lower=lower, strip=strip)
        return val

    if kfrom == "filename":
        val = sample.get("image_path") or sample.get("dicom_id")
        if isinstance(val, str):
            val = _canon_str(val, lower=lower, strip=strip)
        return val

    raise ValueError(f"[ue_keys] Unsupported key.from: {kfrom}")

def collect_keys(dataset, key_spec: Dict[str, Any], *, classwise: bool) -> List[Any]:
    """
    Generic keys collection (for orchestrator union scanning):
    - Recursively handles ConcatDataset
    - classwise: returns "deduplicated class keys" (stable order)
    - samplewise: returns "per-sample key list" (maintains sample order)
    - No int() conversion, preserves raw-key
    """
    # Recursively expand ConcatDataset
    if isinstance(dataset, ConcatDataset):
        out: List[Any] = []
        seen = set()
        for sub in dataset.datasets:
            ks = collect_keys(sub, key_spec, classwise=classwise)
            if classwise:
                for k in ks:
                    if k not in seen:
                        out.append(k); seen.add(k)
            else:
                out.extend(ks)
        return out

    # Non-ConcatDataset: direct iteration
    if classwise:
        seen = set(); uniq: List[Any] = []
        for i in range(len(dataset)):
            sample = dataset[i]
            k = extract_key(sample, i, key_spec)
            if k not in seen:
                uniq.append(k); seen.add(k)
        return uniq
    else:
        return [extract_key(dataset[i], i, key_spec) for i in range(len(dataset))]
