from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import hydra
from omegaconf import DictConfig

from nested_learning.device import resolve_device
from nested_learning.training import run_training_loop, unwrap_config


@hydra.main(config_path="configs", config_name="pilot", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = unwrap_config(cfg)
    device = resolve_device(cfg.train.device)
    run_training_loop(cfg, device=device, distributed=False)


if __name__ == "__main__":
    main()
