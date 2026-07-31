"""Training entry point.

Usage
-----
    python -m src.train --config config/default.yaml [--split random|gene]

What it does
------------
    1.  Fixes all random seeds from config.
    2.  Loads dataset CSV; builds tokenizer.
    3.  Splits using the strategy defined in config (default: gene-level).
    4.  Trains UtrExpressionModel with linear-warmup LR scheduler and early
        stopping on validation MSE.
    5.  Saves:
          - best checkpoint         → results/checkpoints/best.pt
          - training log            → results/training_log.json
          - loss curves plot        → results/plots/loss_curves.png
          - final metrics JSON      → results/metrics_model.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import ExpressionDataset
from src.model import UtrExpressionModel
from src.splits import make_split
from src.utils import (
    load_config, project_root, resolve_device, save_json, set_seed,
)


def build_tokenizer(cfg: dict):
    from multimolecule import RnaTokenizer
    return RnaTokenizer.from_pretrained(
        cfg["model"]["pretrained_name"],
        nmers=cfg["model"]["nmers"],
    )


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
    )


def compute_metrics(preds_log: np.ndarray, trues_log: np.ndarray) -> dict:
    """Compute metrics on both log and original scales."""
    from sklearn.metrics import (
        mean_absolute_percentage_error, mean_squared_error, r2_score,
    )
    preds_orig = np.expm1(preds_log)
    trues_orig = np.expm1(trues_log)
    return {
        "MSE_log": float(mean_squared_error(trues_log, preds_log)),
        "R2_log": float(r2_score(trues_log, preds_log)),
        "R2_orig": float(r2_score(trues_orig, preds_orig)),
        "MAPE_percent_orig": float(
            mean_absolute_percentage_error(trues_orig, preds_orig) * 100
        ),
    }


def evaluate(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tissue_id = batch["tissue_id"].to(device)
            target = batch["expression"].to(device)

            out = model(input_ids, attention_mask, tissue_id)
            total_loss += criterion(out, target).item()

            preds.append(out.cpu().numpy())
            trues.append(target.cpu().numpy())

    return (
        total_loss / len(loader),
        np.concatenate(preds),
        np.concatenate(trues),
    )


def make_plots(history: dict, results_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["train_loss"], label="Train MSE (log)",
            marker="o")
    ax.plot(history["epoch"], history["val_loss"], label="Val MSE (log)",
            marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log target)")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_curves.png", dpi=100)
    plt.close(fig)


def train_one_run(cfg: dict, split_override: str | None = None) -> dict:
    root = project_root()
    set_seed(cfg["seed"])
    device = resolve_device(cfg["training"]["device"])
    print(f"Device: {device}")

    if split_override is not None:
        cfg["split"]["strategy"] = split_override

    # ---------- data ----------
    df = pd.read_csv(root / cfg["data"]["csv_path"])
    print(f"Loaded {len(df)} rows, {df[cfg['data']['gene_column']].nunique()} genes")

    subsets = make_split(df, cfg)
    print(f"Split '{cfg['split']['strategy']}':",
          {k: len(v) for k, v in subsets.items()})

    tokenizer = build_tokenizer(cfg)

    common_kwargs = dict(
        tokenizer=tokenizer,
        utr5_column=cfg["data"]["utr5_column"],
        utr3_column=cfg["data"]["utr3_column"],
        tissue_column=cfg["data"]["tissue_column"],
        target_column=cfg["data"]["target_column"],
        max_seq_len=cfg["model"]["max_seq_len"],
        log1p_target=cfg["data"]["log1p_target"],
    )
    train_ds = ExpressionDataset(subsets["train"], **common_kwargs)
    val_ds = ExpressionDataset(subsets["val"], **common_kwargs)

    train_loader = make_loader(train_ds, cfg["training"]["batch_size"],
                               True, cfg["training"]["num_workers"])
    val_loader = make_loader(val_ds, cfg["training"]["batch_size"],
                             False, cfg["training"]["num_workers"])

    # ---------- model ----------
    model = UtrExpressionModel(
        pretrained_name=cfg["model"]["pretrained_name"],
        n_tissues=cfg["model"]["n_tissues"],
        tissue_embed_dim=cfg["model"]["tissue_embed_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    if cfg["model"]["freeze_encoder_epochs"] > 0:
        model.freeze_encoder(True)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_loader) * cfg["training"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg["training"]["warmup_fraction"] * total_steps),
        num_training_steps=total_steps,
    )
    criterion = nn.MSELoss()

    # ---------- training loop ----------
    results_dir = root / cfg["logging"]["results_dir"]
    (results_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_metrics = None
    epochs_without_improvement = 0
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        if epoch == cfg["model"]["freeze_encoder_epochs"] + 1:
            model.freeze_encoder(False)
            optimizer = AdamW(
                model.parameters(),
                lr=cfg["training"]["learning_rate"],
                weight_decay=cfg["training"]["weight_decay"],
            )
            print(f"[epoch {epoch}] encoder unfrozen")

        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch} train"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tissue_id = batch["tissue_id"].to(device)
            target = batch["expression"].to(device)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask, tissue_id)
            loss = criterion(out, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),
                                     cfg["training"]["gradient_clip_norm"])
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss, preds, trues = evaluate(model, val_loader, criterion, device)
        metrics = compute_metrics(preds, trues)
        print(f"  epoch {epoch}: train={train_loss:.4f}  val={val_loss:.4f}  "
              f"R2_log={metrics['R2_log']:+.4f}  MAPE={metrics['MAPE_percent_orig']:.2f}%")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_metrics = {**metrics, "epoch": epoch}
            if cfg["logging"]["save_best_checkpoint"]:
                torch.save(
                    model.state_dict(),
                    results_dir / "checkpoints" / "best.pt",
                )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg["training"]["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # ---------- save ----------
    save_json(history, results_dir / "training_log.json")
    save_json({
        "split_strategy": cfg["split"]["strategy"],
        "best_val_metrics": best_metrics,
        "final_history": history,
    }, results_dir / "metrics_model.json")

    if cfg["logging"]["save_plots"]:
        make_plots(history, results_dir)

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train UtrExpressionModel")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--split", choices=["random", "gene"], default=None,
                        help="Override split strategy from config")
    args = parser.parse_args()

    cfg_path = project_root() / args.config
    cfg = load_config(cfg_path)
    best = train_one_run(cfg, split_override=args.split)
    print("\nBest validation metrics:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
