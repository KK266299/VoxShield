# file: ue_generate.py
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

# Trigger registries
import src.datasets      # noqa: F401
import src.ue_providers  # noqa: F401

from src.utils.logger import setup_logger, get_logger
from src.core import ExperimentManager
from src.core.ue_orchestrator import generate_training_free


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_dir = HydraConfig.get().runtime.output_dir
    log_file = os.path.join(run_dir, "ue_generate.log")
    logger = setup_logger(log_file=log_file)

    manager = ExperimentManager(cfg)

    kind = str(cfg.get("ue", {}).get("algorithm", {}).get("kind", "")).lower()
    if kind == "training_free":
        # existing training-free implementation: pass clean train/val dataset
        train_ds = manager.build_clean_dataset(split="train")
        val_ds   = manager.build_clean_dataset(split="val")
        ok = generate_training_free(cfg, [train_ds, val_ds])
        if not ok:
            logger.error("[UE] training-free generation did not run (check ue.algorithm/key).")
    elif kind == "training_based":
        manager.setup_model()
        manager.setup_data(mode='train')     # for UE task, return train+val merged loader
        manager.setup_optimizer()            # only build optimizer for model(and multiple surrogate)
        manager.setup_scheduler()
        manager.setup_trainer()              # UE task → UETrainer + noise backend + register export hook
        try:
            manager.train(cfg.training.epochs)
        except Exception as e:
            logger.error(f"[UE] Training failed: {e}")
            raise e
    else:
        logger.error("[UE] No ue.algorithm.kind specified. Nothing to do.")

if __name__ == "__main__":
    main()