import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import shutil
from datetime import datetime
import logging
import warnings
import time

from src.config import RobustnessConfig
from src.plot_style import use_paper_style, use_default_style, use_exploratory_style


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
    warnings.filterwarnings(
        "ignore",
        message="The input matrix is square, but the distance_matrix flag is off.  Did you mean to indicate that this was a distance matrix?",
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

    # Apply plotting style based on config.general.style
    style = getattr(cfg.general, "style")
    print(f"Using style: {style}")
    if style == "paper":
        use_paper_style()
    elif style == "exploratory":
        use_exploratory_style()
    else:
        use_default_style()

    out_dir = bootstrap_output_dir(cfg, args)
    setup_logger(out_dir)
    log_versions(out_dir)

    from src.robustness.pipeline import RobustnessPipeline

    start_time = time.time()
    logging.info("=" * 80)
    logging.info("Starting robustness pipeline")
    logging.info("=" * 80)
    
    pipeline = RobustnessPipeline(cfg, out_dir)
    
    # Prepare data
    step_start = time.time()
    logging.info("\n[Step 1/4] Preparing data...")
    pipeline.prepare_data()
    step_time = time.time() - step_start
    logging.info(f"[Step 1/4] Data preparation completed in {step_time:.2f}s")
    
    # Prepare model
    step_start = time.time()
    logging.info("\n[Step 2/4] Preparing model...")
    pipeline.prepare_model()
    step_time = time.time() - step_start
    logging.info(f"[Step 2/4] Model preparation completed in {step_time:.2f}s")
    
    # Run probes (this is the longest step)
    step_start = time.time()
    logging.info("\n[Step 3/4] Running probes (this may take a while)...")
    summary = pipeline.run_probes()
    step_time = time.time() - step_start
    logging.info(f"[Step 3/4] Probes completed in {step_time:.2f}s ({step_time/60:.1f} minutes)")
    
    # Aggregate and report
    step_start = time.time()
    logging.info("\n[Step 4/4] Aggregating results and generating reports...")
    pipeline.aggregate_and_report(summary)
    step_time = time.time() - step_start
    logging.info(f"[Step 4/4] Reporting completed in {step_time:.2f}s")
    
    total_time = time.time() - start_time
    logging.info("=" * 80)
    logging.info(f"Pipeline completed successfully!")
    logging.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    logging.info("=" * 80)
    print(f"\nRobustness results written to: {out_dir}")
    print(f"Total execution time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()