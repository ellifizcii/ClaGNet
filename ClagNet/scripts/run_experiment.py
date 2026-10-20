"""Command-line runner for CLaGNet and the baseline open-set methods.

Examples
--------
    python scripts/run_experiment.py --method clagnet --setting low --runs 30
    python scripts/run_experiment.py --method all --setting moderate-high --dataset Pollen
"""
import argparse
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (low-openness module, moderate-high-openness module)
FILES = {
    "clagnet": ("clagnet_low.py", "clagnet_moderate_high.py"),
    "arpl": ("arpl_low.py", "arpl_moderate_high.py"),
    "openmax": ("openmax_low.py", "openmax_moderate_high.py"),
    "proser": ("proser_low.py", "proser_moderate_high.py"),
}

# Alternative spellings accepted by --dataset (after normalization).
ALIASES = {"baronm1": "baronmouse", "baron": "baronmouse"}


def normalize(name: str) -> str:
    """Lower-case and strip non-alphanumerics so 'Baron Mouse' == 'baron_mouse' == 'baronm1'."""
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return ALIASES.get(key, key)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run CLaGNet / baseline open-set experiments")
    p.add_argument("--method", choices=[*FILES, "all"], required=True)
    p.add_argument("--setting", choices=["low", "moderate-high"], required=True,
                   help="Openness setting: 'low' (~0.1) or 'moderate-high' (0.20 and 0.30)")
    p.add_argument("--dataset", default="all",
                   help="Dataset name (e.g. Pollen, 'Baron Mouse', baronm1) or 'all'")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument("--runs", type=int, default=30, help="Random class partitions per dataset")
    p.add_argument("--epochs", type=int, default=None, help="Override training epochs (smoke tests)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    if args.runs < 1 or (args.epochs is not None and args.epochs < 1):
        p.error("--runs and --epochs must be positive")
    if not args.data_dir.is_dir():
        p.error(f"Data directory not found: {args.data_dir}")

    methods = list(FILES) if args.method == "all" else [args.method]
    for method in methods:
        filename = FILES[method][args.setting == "moderate-high"]
        path = ROOT / "experiments" / filename
        spec = importlib.util.spec_from_file_location(f"experiments_{method}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        module.DATA_DIR = str(args.data_dir.resolve())
        module.OUTPUT_DIR = str((args.output_dir / method / args.setting).resolve())
        module.N_RUNS = args.runs
        module.DEVICE = args.device
        if args.epochs is not None:
            if hasattr(module, "EPOCHS"):
                module.EPOCHS = args.epochs
            if hasattr(module, "PRETRAIN_EPOCHS"):
                module.PRETRAIN_EPOCHS = args.epochs

        if args.dataset.lower() != "all":
            wanted = normalize(args.dataset)
            module.DATASETS = [d for d in module.DATASETS if normalize(d["name"]) == wanted]
            if not module.DATASETS:
                p.error(f"Unknown dataset '{args.dataset}' for {method}/{args.setting}")

        print(f"\nRunning {method} / {args.setting} ({len(module.DATASETS)} dataset(s))", flush=True)
        module.main()


if __name__ == "__main__":
    main()
