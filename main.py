"""Hydra powered entry point for assembling components."""

import os
import sys

# Ensure src is on path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import hydra
from hydra.core.hydra_config import HydraConfig

from omegaconf import DictConfig, OmegaConf

from src.core import ExperimentManager
from src.utils.logger import setup_logger

# Import modules so they register themselves
import src.datasets  # noqa: F401
import src.models  # noqa: F401
import src.evaluation  # noqa: F401


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_dir = HydraConfig.get().runtime.output_dir
    log_file = os.path.join(run_dir, "train.log")
    logger = setup_logger(log_file=log_file)

    logger.info(f"Running Configs:\n{OmegaConf.to_yaml(cfg)}")

    manager = ExperimentManager(cfg)

    manager.setup_model()
    # todo: test
    manager.setup_data(mode='train')
    manager.setup_optimizer()
    manager.setup_scheduler()
    manager.setup_trainer()

    try:
        manager.train(cfg.training.epochs)
    except Exception as e:
        logger.error(f"[Train] Training failed: {e}")
        raise e


if __name__ == "__main__":
    main()
