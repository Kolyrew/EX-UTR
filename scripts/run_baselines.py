"""Run all baselines under both split strategies and save results as JSON.

Usage
-----
    python scripts/run_baselines.py [--config config/default.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import src.*
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src import baselines
from src.splits import gene_level_split, random_split
from src.utils import load_config, save_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])

    df = pd.read_csv(ROOT / cfg["data"]["csv_path"])
    print(f"Loaded {len(df)} rows, {df['gene_symbol'].nunique()} unique genes")

    common = dict(
        target_col=cfg["data"]["target_column"],
        tissue_col=cfg["data"]["tissue_column"],
        utr5_col=cfg["data"]["utr5_column"],
        utr3_col=cfg["data"]["utr3_column"],
        seed=cfg["seed"],
    )

    val_frac = cfg["split"]["val_fraction"]
    results = {"dataset_size": len(df)}

    for name, split_fn in [
        ("random_split", lambda: random_split(df, val_frac, seed=cfg["seed"])),
        ("gene_level_split",
         lambda: gene_level_split(df, cfg["data"]["gene_column"],
                                  val_frac, seed=cfg["seed"])),
    ]:
        print(f"\n=== {name} ===")
        subsets = split_fn()
        out = baselines.run_all(subsets["train"], subsets["val"], **common)
        for k, m in out.items():
            print(f"  {k:32s}  R²={m['R2']:+.4f}  MSE={m['MSE']:.4f}  "
                  f"MAPE={m['MAPE_percent']:.2f}%")
        results[name] = {
            "train_size": len(subsets["train"]),
            "val_size": len(subsets["val"]),
            **out,
        }

    out_path = ROOT / "results" / "metrics_baselines.json"
    save_json(results, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
