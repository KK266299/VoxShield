# todo: nii.gz
import os
import glob
import argparse
import yaml
import csv
import numpy as np

import nibabel as nib
from nibabel.orientations import aff2axcodes
from scipy.ndimage import zoom
import h5py


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def find_modality_file(case_dir: str, keyword: str) -> str:

    patterns = [
        os.path.join(case_dir, f"*{keyword}*.nii"),
        os.path.join(case_dir, f"*{keyword}*.nii.gz"),
    ]
    for p in patterns:
        files = glob.glob(p)
        if len(files) > 0:
            return files[0]
    raise FileNotFoundError(f"Cannot find modality '{keyword}' in {case_dir}")


def load_nifti_as_canonical(path: str):
    nii = nib.load(path)
    canonical = nib.as_closest_canonical(nii)
    data = canonical.get_fdata(dtype=np.float32)
    affine = canonical.affine
    zooms = canonical.header.get_zooms()[:3]
    axcodes = aff2axcodes(affine)
    return data, affine, zooms, axcodes


def zscore_and_to01_per_modality(
    vol: np.ndarray,
    z_clip: float,
    to_01: bool = True,
) -> np.ndarray:
    brain_mask = vol > 0
    if np.sum(brain_mask) == 0:
        mean = 0.0
        std = 1.0
    else:
        vals = vol[brain_mask]
        mean = float(vals.mean())
        std = float(vals.std())
        if std < 1e-6:
            std = 1.0

    vol_z = np.zeros_like(vol, dtype=np.float32)
    vol_z[brain_mask] = (vol[brain_mask] - mean) / std

    # clip
    vol_z = np.clip(vol_z, -z_clip, z_clip)

    if not to_01:
        return vol_z

    # [-z_clip, z_clip] -> [0,1]
    vol_01 = (vol_z + z_clip) / (2.0 * z_clip)
    vol_01 = np.clip(vol_01, 0.0, 1.0)
    return vol_01


def resize_volume(
    vol: np.ndarray,
    target_shape: tuple,
    is_label: bool = False,
) -> np.ndarray:
    if vol.ndim == 4:
        # (C, H, W, D)
        c, h, w, d = vol.shape
        th, tw, td = target_shape
        zoom_factors = (1.0, th / h, tw / w, td / d)
        order = 0 if is_label else 3
        resized = zoom(vol, zoom_factors, order=order)
    elif vol.ndim == 3:
        # (H, W, D)
        h, w, d = vol.shape
        th, tw, td = target_shape
        zoom_factors = (th / h, tw / w, td / d)
        order = 0 if is_label else 3
        resized = zoom(vol, zoom_factors, order=order)
    else:
        raise ValueError(f"Unsupported volume ndim={vol.ndim}")
    return resized.astype(vol.dtype)


def remap_labels(seg: np.ndarray, mapping: dict) -> np.ndarray:
    """
    seg: 3D label map
    mapping: dict, e.g. {0:0, 1:1, 2:2, 4:3}
    """
    seg_remap = np.zeros_like(seg, dtype=np.uint8)
    for src, dst in mapping.items():
        seg_remap[seg == int(src)] = int(dst)
    return seg_remap


def find_case_dir_by_id(raw_root: str, case_id: str) -> str:
    for root, dirs, files in os.walk(raw_root):
        if case_id in dirs:
            return os.path.join(root, case_id)
    raise FileNotFoundError(f"Cannot find case dir for ID {case_id} under {raw_root}")


def load_cases_from_mapping(mapping_csv: str, raw_root: str):
    cases = []
    with open(mapping_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grade = row.get("Grade", "")
            case_id = row.get("BraTS_2019_subject_ID", "")
            if case_id is None or case_id.strip() == "" or case_id.upper() == "NA":
                continue
            case_id = case_id.strip()
            grade = grade.strip() if grade is not None else ""
            case_dir = find_case_dir_by_id(raw_root, case_id)
            cases.append(
                {
                    "case_id": case_id,
                    "grade": grade,
                    "case_dir": case_dir,
                }
            )
    return cases


def process_case(
    case_dir: str,
    modalities: list,
    label_remap: dict,
    target_shape: tuple,
    z_clip: float,
    to_01: bool,
    out_dir: str,
):

    case_id = os.path.basename(case_dir.rstrip("/"))

    img_list = []
    canonical_affine = None
    orig_shape = None
    orig_spacing = None

    for m in modalities:
        path = find_modality_file(case_dir, m)
        vol, affine, zooms, axcodes = load_nifti_as_canonical(path)

        if canonical_affine is None:
            canonical_affine = affine
            orig_shape = vol.shape
            orig_spacing = zooms
        else:
            if vol.shape != orig_shape:
                raise ValueError(
                    f"Modality {m} shape {vol.shape} != {orig_shape} in case {case_id}"
                )

        vol_norm = zscore_and_to01_per_modality(vol, z_clip=z_clip, to_01=to_01)
        img_list.append(vol_norm)

    image = np.stack(img_list, axis=0).astype(np.float32)  # (C, H, W, D)

    seg_path = find_modality_file(case_dir, "seg")
    seg_vol, seg_affine, _, _ = load_nifti_as_canonical(seg_path)
    seg_vol = seg_vol.astype(np.int16)

    if seg_vol.shape != orig_shape:
        raise ValueError(
            f"Seg shape {seg_vol.shape} != image shape {orig_shape} in case {case_id}"
        )

    image_resized = resize_volume(image, target_shape=target_shape, is_label=False)
    seg_resized = resize_volume(seg_vol, target_shape=target_shape, is_label=True)

    seg_remap = remap_labels(seg_resized, label_remap)

    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{case_id}.h5")

    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "image",
            data=image_resized.astype(np.float32),
            compression="gzip",
        )
        f.create_dataset(
            "label",
            data=seg_remap.astype(np.uint8),
            compression="gzip",
        )
        f.attrs["case_id"] = case_id
        f.attrs["orig_shape"] = np.array(orig_shape, dtype=np.int32)
        f.attrs["orig_spacing"] = np.array(orig_spacing, dtype=np.float32)
        f.attrs["target_shape"] = np.array(target_shape, dtype=np.int32)
        f.attrs["z_clip"] = float(z_clip)
        f.attrs["to_01"] = int(to_01)

    return case_id, out_path


def split_cases(case_ids, split_ratio, seed=42):
    ratios = np.array(split_ratio, dtype=float)
    ratios = ratios / ratios.sum()
    r_train, r_val, r_test = ratios.tolist()

    n = len(case_ids)
    np.random.seed(seed)
    idx = np.random.permutation(n)

    n_train = int(round(r_train * n))
    n_val = int(round(r_val * n))
    if n_train + n_val > n:
        n_val = n - n_train
    n_test = n - n_train - n_val

    splits = {}
    for i, j in enumerate(idx):
        cid = case_ids[j]
        if i < n_train:
            splits[cid] = "train"
        elif i < n_train + n_val:
            splits[cid] = "val"
        else:
            splits[cid] = "test"
    return splits


def build_index_csv(csv_path: str, records: list, splits: dict):
    fieldnames = ["case_id", "grade", "split", "volume_path", "label_path"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            cid = r["case_id"]
            row = {
                "case_id": cid,
                "grade": r.get("grade", ""),
                "split": splits.get(cid, "train"),
                "volume_path": r["h5_path"],
                "label_path": r["h5_path"],
            }
            writer.writerow(row)


def build_split_csvs(root_dir: str, records: list, splits: dict):
    fieldnames = ["case_id", "grade", "volume_path", "label_path"]

    grouped = {"train": [], "val": [], "test": []}
    for r in records:
        cid = r["case_id"]
        split = splits.get(cid, "train")
        if split not in grouped:
            continue
        grouped[split].append(
            {
                "case_id": cid,
                "grade": r.get("grade", ""),
                "volume_path": r["h5_path"],
                "label_path": r["h5_path"],
            }
        )

    for split_name, rows in grouped.items():
        csv_path = os.path.join(root_dir, f"{split_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"{split_name}.csv saved to: {csv_path} (n={len(rows)})")


def main(config_path: str):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]

    raw_root = data_cfg["raw_root"]
    mapping_csv = data_cfg["name_mapping_csv"]
    preproc_root = data_cfg["preproc_root"]

    modalities = data_cfg["modalities"]
    label_remap = {int(k): int(v) for k, v in data_cfg["label_remap"].items()}
    target_shape = tuple(int(x) for x in data_cfg["target_shape"])
    z_clip = float(data_cfg.get("z_clip", 5.0))
    to_01 = bool(data_cfg.get("to_01", True))

    split_ratio = data_cfg.get("split_ratio", [0.7, 0.15, 0.15])
    split_seed = int(data_cfg.get("split_seed", 42))

    run_preprocess = bool(data_cfg.get("run_preprocess", False))

    ensure_dir(preproc_root)
    out_h5_dir = os.path.join(preproc_root, "h5")
    ensure_dir(out_h5_dir)

    cases = load_cases_from_mapping(mapping_csv, raw_root)
    print(f"Found {len(cases)} cases from mapping CSV: {mapping_csv}")
    if not run_preprocess:
        print(
            "[INFO] run_preprocess=False, skipping preprocessing and assuming H5 files already exist. "
        )

    records = []
    for idx, c in enumerate(cases):
        case_id = c["case_id"]
        grade = c["grade"]
        case_dir = c["case_dir"]
        print(f"[{idx+1}/{len(cases)}] Processing {case_id} (grade={grade}) ...")

        if run_preprocess:
            case_id2, h5_path = process_case(
                case_dir=case_dir,
                modalities=modalities,
                label_remap=label_remap,
                target_shape=target_shape,
                z_clip=z_clip,
                to_01=to_01,
                out_dir=out_h5_dir,
            )
            assert case_id2 == case_id
        else:
            h5_path = os.path.join(out_h5_dir, f"{case_id}.h5")
            if not os.path.exists(h5_path):
                raise FileNotFoundError(
                    f"[ERROR] run_preprocess=False but not found H5 files: {h5_path}\n"
                )

        records.append({"case_id": case_id, "h5_path": h5_path, "grade": grade})

    case_ids = [r["case_id"] for r in records]
    splits = split_cases(case_ids, split_ratio=split_ratio, seed=split_seed)

    build_split_csvs(preproc_root, records, splits)

    print("Done.")
    print(f"H5 files dir: {out_h5_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess BraTS19 to 3D HDF5 volumes"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config"
    )
    args = parser.parse_args()
    main(args.config)
