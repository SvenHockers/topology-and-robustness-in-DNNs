import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import shutil
from datetime import datetime
import logging
import warnings

from src.config import RobustnessConfig


def parse_args():
    p = argparse.ArgumentParser(description="Run robustness pipeline with YAML configuration")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--output_dir", default=None, help="Override output directory")
    p.add_argument("--checkpoint", default=None, help="Optional model checkpoint to load")
    p.add_argument("--exp_name", default=None, help="Override experiment name")
    p.add_argument("--sample_limit", type=int, default=None, help="Limit number of validation samples")
    return p.parse_args()


def bootstrap_output_dir(cfg: RobustnessConfig, cli_args) -> str:
    out_dir = cli_args.output_dir or cfg.general.output_dir
    if cli_args.exp_name:
        out_dir = os.path.join(os.path.dirname(out_dir), cli_args.exp_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    # copy YAML
    shutil.copy2(cli_args.config, os.path.join(out_dir, "config.yaml"))
    # write resolved config as JSON for quick view
    with open(os.path.join(out_dir, "resolved_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, default=lambda o: o.__dict__, indent=2)
    return out_dir


def setup_logger(out_dir: str):
    log_path = os.path.join(out_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def log_versions(out_dir: str):
    import torch, numpy, ripser, persim  # noqa: F401
    versions = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "ripser": getattr(ripser, "__version__", "unknown"),
        "persim": getattr(persim, "__version__", "unknown"),
    }
    with open(os.path.join(out_dir, "versions.json"), "w") as f:
        json.dump(versions, f, indent=2)


def main():
    warnings.filterwarnings(
        "ignore",
        message="The input point cloud has more columns than rows; did you mean to transpose?",
        module="ripser.ripser",
    )

    args = parse_args()
    cfg = RobustnessConfig.from_yaml(args.config)
    # CLI overrides
    if args.checkpoint:
        cfg.model.checkpoint = args.checkpoint
    if args.sample_limit is not None:
        cfg.general.sample_limit = args.sample_limit
    if args.exp_name:
        cfg.general.exp_name = args.exp_name
    if args.output_dir:
        cfg.general.output_dir = args.output_dir

    out_dir = bootstrap_output_dir(cfg, args)
    setup_logger(out_dir)
    log_versions(out_dir)

    from src.robustness.pipeline import RobustnessPipeline

    pipeline = RobustnessPipeline(cfg, out_dir)
    pipeline.prepare_data()
    pipeline.prepare_model()
    summary = pipeline.run_probes()
    pipeline.aggregate_and_report(summary)
    print(f"Robustness results written to: {out_dir}")


if __name__ == "__main__":
    main()


