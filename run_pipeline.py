from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_STAGE_ORDER = [
    "preprocess",
    "classifier",
    "regressor",
    "evaluate",
    "visualize",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the rainfall prediction pipeline end-to-end or stage by stage."
    )
    parser.add_argument(
        "stages",
        nargs="*",
        choices=["all", *DEFAULT_STAGE_ORDER],
        help=(
            "Stages to run in the order provided. "
            "Use 'all' or omit stages to run the full pipeline."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML config file. Defaults to configs/config.yaml.",
    )
    return parser.parse_args()


def resolve_stages(requested_stages):
    if not requested_stages or "all" in requested_stages:
        return DEFAULT_STAGE_ORDER

    seen = set()
    ordered = []
    for stage in requested_stages:
        if stage not in seen:
            ordered.append(stage)
            seen.add(stage)
    return ordered


def run_stage(stage, config_path):
    stage_handlers = {
        "preprocess": ("preprocess", "run_preprocessing"),
        "classifier": ("train_classifier", "run_training"),
        "regressor": ("train_regressor", "run_training"),
        "evaluate": ("evaluate", "evaluate_all"),
        "visualize": ("visualize", "run_all_eda_plots"),
    }

    log.info("=" * 60)
    log.info("Running stage: %s", stage)
    module_name, function_name = stage_handlers[stage]
    module = importlib.import_module(module_name)
    getattr(module, function_name)(config_path=config_path)


def main():
    args = parse_args()
    stages = resolve_stages(args.stages)

    for stage in stages:
        run_stage(stage, args.config)

    log.info("=" * 60)
    log.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
