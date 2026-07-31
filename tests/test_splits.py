"""Smoke tests for split strategies."""
from __future__ import annotations

import pandas as pd
import pytest

from src.splits import gene_level_split, random_split


@pytest.fixture
def toy_df():
    """5 genes, 3 tissues each => 15 rows."""
    genes = [f"G{i}" for i in range(1, 6)]
    tissues = ["A", "B", "C"]
    rows = []
    for g in genes:
        for t in tissues:
            rows.append({"gene_symbol": g, "tissue": t, "expression_level": 1.0})
    return pd.DataFrame(rows)


def test_random_split_sizes(toy_df):
    subs = random_split(toy_df, val_fraction=0.2, seed=42)
    assert len(subs["train"]) + len(subs["val"]) == len(toy_df)


def test_gene_split_no_overlap(toy_df):
    subs = gene_level_split(toy_df, val_fraction=0.2, seed=42)
    train_genes = set(subs["train"]["gene_symbol"].unique())
    val_genes = set(subs["val"]["gene_symbol"].unique())
    # The key invariant: no gene appears in both train and val
    assert train_genes.isdisjoint(val_genes)


def test_gene_split_covers_all_rows(toy_df):
    subs = gene_level_split(toy_df, val_fraction=0.2, seed=42)
    assert len(subs["train"]) + len(subs["val"]) == len(toy_df)


def test_split_is_reproducible(toy_df):
    a = gene_level_split(toy_df, val_fraction=0.2, seed=42)
    b = gene_level_split(toy_df, val_fraction=0.2, seed=42)
    assert set(a["val"]["gene_symbol"]) == set(b["val"]["gene_symbol"])
