# file: src/core/trainers/ue_trainer.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Iterable, Tuple, Set

import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from ..trainer_base import TrainerBase, HookBase
from ...utils.logger import get_logger
from ...utils.config import get_config, require_config
from ...registry import get_plugin
from .. import ue_algos as _ue_algos  # noqa: F401
from ...core.ue_keys import collect_keys

class UETrainer(TrainerBase):
    """
    UE Trainer (dual loop):
      - Each epoch = S-step (fixed number of batches, train surrogate) + N-step (full pass, update noise)
      - Plugin provides surrogate_step_batch(...) / noise_step_batch(...)
      - Export is handled by a pure export Hook (after_train)
    """
    def __init__(self, config, device, evaluation_strategy=None):
        super().__init__(config, device)
        self.evaluation_strategy = evaluation_strategy
        self.plugin = None
        self._is_last_epoch: bool = False

        self.opt_surrogates: Dict[str, torch.optim.Optimizer] = {}
        self.surrogates: Dict[str, torch.nn.Module] = {}

        # export context
        self._train_keys: List[Any] = []     
        self._val_keys: List[Any] = []      
        self._perturb_type: str = "samplewise"
        self._eps = require_config(self.config, "ue.algorithm.params.epsilon")
        self._export_cfg = require_config(self.config, "ue.io")

        self.logger = get_logger()

    # ---------------------- lifecycle ---------------------- #
    def setup(
        self,
        model: torch.nn.Module,
        criterion: Optional[torch.nn.Module],
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        evaluation_strategy=None,
        *,
        plugin_name: Optional[str] = None,
        noise_backend: Optional[Any] = None,
        surrogates: Optional[Dict[str, torch.nn.Module]] = None,
        sur_optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    ):
        super().setup(model, criterion, optimizer, scheduler, evaluation_strategy)

        algo_name = require_config(self.config, "ue.algorithm.name")
        plugin_cls = get_plugin(algo_name)
        self.plugin = plugin_cls()

        self.surrogates = dict(surrogates or {})
        self.opt_surrogates = dict(sur_optimizers or {})

        self.noise_backend = noise_backend
        self._perturb_type = str(get_config(self.config, "ue.key.type", "samplewise")).lower()

        if bool(get_config(self.config, "ue.io.enabled", True)):
            self.register_hooks([_UEExportHook(self)])

    # ---------------------- core loop ---------------------- #
    def train_epoch(self, epoch: int, data_loader: DataLoader) -> Dict[str, float]:
        self._is_last_epoch = epoch == self._max_epochs - 1
        if torch.distributed.is_initialized():
            sampler = getattr(data_loader, 'sampler', None)
            if sampler is not None and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)

        sur_train_step = int(require_config(self.config, "ue.algorithm.params.surrogate_step"))

        # Hooks
        for h in self._hooks:
            h.before_train_epoch()

        # ---------- Surrogate-step：Train surrogate model for several batches ----------
        metrics = self._init_epoch_metrics()
        show_pbar = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        it = iter(data_loader)
        pbar_sur = tqdm(range(sur_train_step), desc=f"Epoch {epoch} [SUR]", leave=False) if show_pbar else range(sur_train_step)
        sur_loss_sum, sur_loss_cnt = 0.0, 0

        for _ in pbar_sur:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(data_loader)
                batch = next(it)

            for h in self._hooks:
                h.before_train_step()
            self.before_step()

            # call the surrogate-step of the plugin
            if hasattr(self.plugin, "surrogate_step_batch"):
                step_metrics = self.plugin.surrogate_step_batch(self, batch)
                sur_loss_sum += float(step_metrics.get("surrogate_loss", 0.0))
                sur_loss_cnt += 1
            else:
                raise RuntimeError("UE plugin must implement surrogate_step_batch(trainer, batch).")

            self._update_metrics(metrics, step_metrics)
            if show_pbar:
                pbar_sur.set_postfix({"surrogate_loss": step_metrics.get("surrogate_loss", 0.0)})

            self.after_step()
            for h in self._hooks:
                h.after_train_step()
            self.iter += 1

        if sur_loss_cnt > 0:
            self.logger.info(f"[SUR] avg_surrogate_loss={sur_loss_sum / sur_loss_cnt:.4f}")

        # ---------- Noise-step：Update noise for one epoch ----------
        noise_loss_sum, noise_loss_cnt = 0.0, 0
        delta_linf_max = 0.0
        pbar_noise = tqdm(data_loader, desc=f"Epoch {epoch} [NOISE]", leave=False) if show_pbar else data_loader
        for batch in pbar_noise:
            if hasattr(self.plugin, "noise_step_batch"):
                step_metrics = self.plugin.noise_step_batch(self, batch)
                noise_loss_sum += float(step_metrics.get("noise_loss", 0.0))
                noise_loss_cnt += 1
                delta_linf_max = max(delta_linf_max, float(step_metrics.get("delta_linf", 0.0)))
            else:
                raise RuntimeError("UE plugin must implement noise_step_batch(trainer, batch).")

        if noise_loss_cnt > 0:
            self.logger.info(
                f"[NOISE] avg_noise_loss={noise_loss_sum / noise_loss_cnt:.4f} | "
                f"max||δ||∞={delta_linf_max:.4f}"
            )
        # Hooks
        for h in self._hooks:
            h.after_train_epoch()

        return self._finalize_epoch_metrics(metrics)

    def run_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        raise RuntimeError("UE plugin must implement surrogate_step_batch(trainer, batch).")

    def _is_best_model(self, eval_stats: Dict[str, float]) -> bool:
        return False

    # ---------------------- helpers for export ---------------------- #
    def _collect_keys_for_export(self, train_loader: DataLoader):
        self._train_keys.clear()
        self._val_keys.clear()

        key_spec = get_config(self.config, "ue.key", {})
        ktype = str(get_config(key_spec, "type", "samplewise")).lower()
        classwise = (ktype == "classwise")

        ds = getattr(train_loader, "dataset", None)
        if isinstance(ds, ConcatDataset) and len(ds.datasets) == 2:
            ds_tr, ds_va = ds.datasets[0], ds.datasets[1]
            self._train_keys = collect_keys(ds_tr, key_spec, classwise=classwise)
            self._val_keys   = collect_keys(ds_va, key_spec, classwise=classwise)
        else:
            self._train_keys = collect_keys(ds, key_spec, classwise=classwise)
            self._val_keys = []

    def train(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader = None, eval_on_train: bool = False) -> Dict[str, List]:
        try:
            self._collect_keys_for_export(train_loader)
        except Exception as e:
            self.logger.warning(f"[UE] failed to pre-collect keys for export: {e}")
        self._max_epochs = int(epochs)
        self.train_loader = train_loader
        return super().train(epochs, train_loader, val_loader, test_loader, eval_on_train)


class _UEExportHook(HookBase):
    def __init__(self, trainer: UETrainer):
        self.trainer = trainer

    def after_train_epoch(self):
        t = self.trainer
        cfg = t.config

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        if not bool(get_config(cfg, "ue.io.enabled", True)):
            return

        epoch_idx = int(getattr(t, "epoch", 0))
        start_ep  = int(get_config(cfg, "ue.io.save_from_epoch", 50))
        every     = int(get_config(cfg, "ue.io.save_every", 10))
        is_last   = bool(getattr(t, "_is_last_epoch", False))

        if not is_last:
            if epoch_idx < start_ep or ((epoch_idx - start_ep) % max(every, 1) != 0):
                return
        else:
            t.logger.info(f"[UE] [epoch {epoch_idx}] Last epoch -> force export")

        try:
            from ..ue_artifacts import write_shards, write_files
        except Exception as e:
            t.logger.error(f"[UE] Cannot import ue_export helpers: {e}")
            return

        import os, json

        store_dir_root = str(require_config(cfg, "ue.store_dir"))
        store_dir = os.path.join(store_dir_root, f"epoch_{epoch_idx:04d}")
        os.makedirs(store_dir, exist_ok=True)

        strategy  = str(get_config(cfg, "ue.io.strategy", "shards")).lower()
        shard_sz  = int(get_config(cfg, "ue.io.shard_size", 1024))
        eps       = float(get_config(cfg, "ue.algorithm.params.epsilon", t._eps))
        pert_type = t._perturb_type

        key_cfg = get_config(t.config, "ue.key", {})
        key_spec = {
            "namespace": get_config(key_cfg, "namespace", ""),
            "source": {
                "type": get_config(key_cfg, "from", "field"),
                "name": get_config(key_cfg, "field", "subject_id"),
            },
            "canonicalization": {
                "lower": bool(get_config(key_cfg, "lower", True)),
                "strip": bool(get_config(key_cfg, "strip", True)),
            },
        }

        if t.noise_backend is None:
            t.logger.warning("[UE] No noise_backend bound; export skipped.")
            return

        try:
            keys_all = t.noise_backend.keys_raw()
        except Exception:
            keys_all = list(t._train_keys) + list(t._val_keys)

        def _gather(keys: List[Any]) -> List[Tuple[Any, torch.Tensor]]:
            out: List[Tuple[Any, torch.Tensor]] = []
            for k in keys:
                try:
                    n = t.noise_backend.get_noise(k, pert_type)
                    out.append((k, n))
                except Exception as e:
                    t.logger.warning(f"[UE] fail to fetch noise for key={k}: {e}")
            return out

        entries = _gather(keys_all)
        if not entries:
            t.logger.warning("[UE] No entries to export.")
            return

        try:
            if strategy == "files":
                base_manifest = write_files(store_dir, entries, eps=eps, perturb_type=pert_type, key_spec=key_spec)
            else:
                base_manifest = write_shards(store_dir, entries, eps=eps, shard_size=shard_sz, perturb_type=pert_type, key_spec=key_spec)
            t.logger.info(f"[UE] [epoch {epoch_idx}] Base manifest written at: {base_manifest}")
        except Exception as e:
            t.logger.error(f"[UE] Export failed when writing base manifest: {e}")
            return

        if bool(get_config(cfg, "ue.io.split_manifests", True)) and (t._train_keys or t._val_keys):
            try:
                with open(base_manifest, "r") as f:
                    base = json.load(f)

                def _subset_manifest(raw_keys_set: set, subdir: str) -> str:
                    sub_root = os.path.join(os.path.dirname(base_manifest), subdir)
                    os.makedirs(sub_root, exist_ok=True)
                    sub = dict(base)
                    sub["entries"] = []
                    for en in base["entries"]:
                        keep = [k for k in en["keys"] if k in raw_keys_set]
                        if keep:
                            sub["entries"].append({"path": en["path"], "keys": keep})
                    sub_path = os.path.join(sub_root, "manifest.json")
                    with open(sub_path, "w") as f:
                        json.dump(sub, f, indent=2)
                    return sub_path

                tr_path = _subset_manifest(set(t._train_keys), "train")
                va_path = _subset_manifest(set(t._val_keys), "val")
                t.logger.info(f"[UE] [epoch {epoch_idx}] Split manifests -> train:{tr_path} | val:{va_path}")
            except Exception as e:
                t.logger.warning(f"[UE] Failed to write split manifests: {e}")