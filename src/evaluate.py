"""Evaluate a trained checkpoint and produce diagnostic plots.

Usage
-----
    python -m src.evaluate --config config/default.yaml \\
                           --checkpoint results/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import ExpressionDataset
from src.model import UtrExpressionModel
from src.splits import make_split
from src.utils import (
    load_config, project_root, resolve_device, save_json, set_seed,
)


def scatter_true_vs_pred(trues_orig, preds_orig, out_path: Path) -> None:
    from sklearn.metrics import r2_score
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(trues_orig, preds_orig, s=8, alpha=0.35, color="#1F3A5F")
    lo = min(trues_orig.min(), preds_orig.min())
    hi = max(trues_orig.max(), preds_orig.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#B25C48", lw=1)
    ax.set_xlabel("True expression (original scale)")
    ax.set_ylabel("Predicted expression")
    ax.set_title(f"True vs predicted   R² = {r2_score(trues_orig, preds_orig):.3f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def r2_per_tissue(val_df: pd.DataFrame, preds_orig: np.ndarray,
                  tissue_col: str, target_col: str, out_path: Path) -> dict:
    from sklearn.metrics import r2_score

    df = val_df.copy()
    df["pred"] = preds_orig
    tissue_r2 = {}
    for t, sub in df.groupby(tissue_col):
        if len(sub) >= 3:
            tissue_r2[t] = float(r2_score(sub[target_col].values, sub["pred"].values))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    tissues = list(tissue_r2.keys())
    values = [tissue_r2[t] for t in tissues]
    colors = ["#1F3A5F" if v > 0 else "#B25C48" for v in values]
    ax.bar(tissues, values, color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticklabels(tissues, rotation=45, ha="right")
    ax.set_ylabel(r"R$^2$")
    ax.set_title("Validation R² by tissue")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    return tissue_r2


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["random", "gene"], default=None)
    args = parser.parse_args()

    root = project_root()
    cfg = load_config(root / args.config)
    if args.split is not None:
        cfg["split"]["strategy"] = args.split

    set_seed(cfg["seed"])
    device = resolve_device(cfg["training"]["device"])

    df = pd.read_csv(root / cfg["data"]["csv_path"])
    subsets = make_split(df, cfg)
    val_df = subsets["val"]

    from multimolecule import RnaTokenizer
    tokenizer = RnaTokenizer.from_pretrained(
        cfg["model"]["pretrained_name"], nmers=cfg["model"]["nmers"],
    )

    ds = ExpressionDataset(
        val_df, tokenizer=tokenizer,
        utr5_column=cfg["data"]["utr5_column"],
        utr3_column=cfg["data"]["utr3_column"],
        tissue_column=cfg["data"]["tissue_column"],
        target_column=cfg["data"]["target_column"],
        max_seq_len=cfg["model"]["max_seq_len"],
        log1p_target=cfg["data"]["log1p_target"],
    )
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                        shuffle=False, num_workers=cfg["training"]["num_workers"])

    model = UtrExpressionModel(
        pretrained_name=cfg["model"]["pretrained_name"],
        n_tissues=cfg["model"]["n_tissues"],
        tissue_embed_dim=cfg["model"]["tissue_embed_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    preds_log, trues_log = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        batch["tissue_id"].to(device))
            preds_log.append(out.cpu().numpy())
            trues_log.append(batch["expression"].cpu().numpy())

    preds_log = np.concatenate(preds_log)
    trues_log = np.concatenate(trues_log)
    preds_orig = np.expm1(preds_log)
    trues_orig = np.expm1(trues_log)

    from sklearn.metrics import (
        mean_absolute_percentage_error, mean_squared_error, r2_score,
    )
    metrics = {
        "split_strategy": cfg["split"]["strategy"],
        "n_val": len(val_df),
        "MSE_log": float(mean_squared_error(trues_log, preds_log)),
        "R2_log": float(r2_score(trues_log, preds_log)),
        "R2_orig": float(r2_score(trues_orig, preds_orig)),
        "MAPE_percent_orig": float(
            mean_absolute_percentage_error(trues_orig, preds_orig) * 100
        ),
    }
    print(json.dumps(metrics, indent=2))

    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    scatter_true_vs_pred(trues_orig, preds_orig,
                         plots_dir / "scatter_true_vs_pred.png")
    per_tissue = r2_per_tissue(val_df, preds_orig,
                               cfg["data"]["tissue_column"],
                               cfg["data"]["target_column"],
                               plots_dir / "r2_by_tissue.png")

    metrics["r2_by_tissue"] = per_tissue
    save_json(metrics, root / "results" / f"eval_{cfg['split']['strategy']}.json")


if __name__ == "__main__":
    main()
