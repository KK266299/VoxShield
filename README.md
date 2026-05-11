# Voxshield

**Voxshield** is a UNet based deep learning framework for 3D medical image **segmentation** and **unlearnable example (UE) generation**, supporting BraTS19 (brain tumor) and FLARE21 (abdominal organ) benchmarks.

---

## Project Structure

```
Voxshield/
├── main.py                        # Segmentation / victim training entry point
├── ue_generate.py                 # UE generation entry point
├── requirements.txt               # Dependencies
├── configs/                       # Hydra configuration files
│   ├── config.yaml
│   ├── _global_patches/
│   ├── dataset/
│   ├── method/
│   ├── task/
│   └── training/
├── scripts/                       # Data preparation scripts
│   ├── prepare_brats19.py
│   └── prepare_flare21.py
└── src/
    ├── core/
    │   ├── trainers/
    │   └── ue_algos/
    │       └── noise_slice_frequence_learnable.py
    ├── datasets/
    ├── models/
    ├── ue_providers/
    └── utils/
```

---

## Reproduction

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# BraTS19
python scripts/prepare_brats19.py --config scripts/configs/brats19.yaml

# FLARE21
python scripts/prepare_flare21.py --config scripts/configs/flare21.yaml
```

### 3. Generate Unlearnable Examples

**BraTS19**

```bash
python ue_generate.py \
    dataset=brats19 \
    task=brats19_ue \
    method=noise_slice_frequence_learnable \
    training.epochs=100 \
    training.batch_size=8 \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.noise_step=1 \
    ue.algorithm.params.surrogate_step=10 \
    ue.algorithm.params.roi_aware=true \
    ue.algorithm.params.soft_edge=false \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.surrogates.s_seg.in_channels=4 \
    ue.surrogates.s_seg.num_classes=4 \
    training.gpu_ids=[4] \
    task.run_name=nofreq_learnable_zdiv02_logits005 \
    ue.algorithm.params.z_diversity_weight=0.2 \
    ue.algorithm.params.logits_div_enabled=true \
    ue.algorithm.params.logits_div_weight=0.05
```

**FLARE21**

```bash
python ue_generate.py \
    dataset=flare21 \
    task=flare21_ue \
    method=noise_slice_frequence_learnable \
    training.epochs=100 \
    training.batch_size=8 \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.noise_step=1 \
    ue.algorithm.params.surrogate_step=10 \
    ue.algorithm.params.roi_aware=true \
    ue.algorithm.params.soft_edge=false \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.surrogates.s_seg.in_channels=1 \
    ue.surrogates.s_seg.num_classes=5 \
    training.gpu_ids=[4] \
    task.run_name=nofreq_learnable_zdiv02_logits005 \
    ue.algorithm.params.z_diversity_weight=0.2 \
    ue.algorithm.params.logits_div_enabled=true \
    ue.algorithm.params.logits_div_weight=0.05
```

### 4. Train Victim Model on Unlearnable Examples

After UE generation, update `manifest_path` to the saved `manifest.json` and run:

```bash
python main.py \
    method=poison_files \
    model.pretrained=false \
    dataset=brats19 \
    task=brats19_seg \
    task.run_name=victim_nofreq_learnable_zdiv02_logits005 \
    model=unet \
    model.name=unet \
    training.epochs=100 \
    training.optimizer=adam \
    training.optimizers.adam.lr=5e-4 \
    training.gpu_ids=[4] \
    training.batch_size=8 \
    training.eval_batch_size=8 \
    training.data.poison.perturb_type=samplewise \
    training.data.poison.key.type=samplewise \
    training.data.poison.key.from=field \
    training.data.poison.key.field=case_id \
    training.data.poison.source.type=manifest \
    training.data.poison.source.manifest_path=/path/to/noise/epoch_0099/manifest.json
```

---

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- CUDA recommended

See `requirements.txt` for the full dependency list.

---

## Citation

If you use this project, please cite the corresponding work.
