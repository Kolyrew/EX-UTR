"""Dataset split strategies.

Two strategies are implemented:

* ``random_split``     — split by rows. Same gene may appear in both train and
                          validation (with different tissues). Optimistic
                          metrics due to data leakage; kept for reproducing
                          historical numbers.
* ``gene_level_split`` — split by unique gene identifiers. A gene is either
                          entirely in train or entirely in validation. This is
                          the correct strategy when the goal is to predict
                          expression for unseen genes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_split(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Row-level random split (kept for reproducing historical numbers)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))

    n_val = int(val_fraction * len(df))
    n_test = int(test_fraction * len(df))

    val_idx = idx[:n_val]
    test_idx = idx[n_val : n_val + n_test]
    train_idx = idx[n_val + n_test:]

    out = {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
    }
    if n_test > 0:
        out["test"] = df.iloc[test_idx].reset_index(drop=True)
    return out


def gene_level_split(
    df: pd.DataFrame,
    gene_column: str = "gene_symbol",
    val_fraction: float = 0.1,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split by unique gene IDs. No gene appears in more than one subset."""
    rng = np.random.default_rng(seed)
    genes = df[gene_column].unique()
    perm = rng.permutation(len(genes))
    genes = genes[perm]

    n_val = int(val_fraction * len(genes))
    n_test = int(test_fraction * len(genes))

    val_genes = set(genes[:n_val])
    test_genes = set(genes[n_val : n_val + n_test])

    train_df = df[~df[gene_column].isin(val_genes | test_genes)].reset_index(drop=True)
    val_df = df[df[gene_column].isin(val_genes)].reset_index(drop=True)

    out = {"train": train_df, "val": val_df}
    if n_test > 0:
        out["test"] = df[df[gene_column].isin(test_genes)].reset_index(drop=True)
    return out


def make_split(df: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """Dispatch to the right split strategy based on config."""
    strategy = config["split"]["strategy"]
    val_frac = config["split"]["val_fraction"]
    test_frac = config["split"].get("test_fraction", 0.0)
    seed = config.get("seed", 42)

    if strategy == "random":
        return random_split(df, val_frac, test_frac, seed=seed)
    elif strategy == "gene":
        return gene_level_split(
            df,
            gene_column=config["data"]["gene_column"],
            val_fraction=val_frac,
            test_fraction=test_frac,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown split strategy: {strategy!r}")
