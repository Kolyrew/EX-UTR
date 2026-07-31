"""Smoke tests for baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import baselines


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "gene_symbol": [f"G{i//4}" for i in range(n)],
        "tissue": rng.choice(["A", "B", "C"], size=n),
        "UTR5_Sequence": ["".join(rng.choice(list("ACGT"), 100)) for _ in range(n)],
        "UTR3_Sequence": ["".join(rng.choice(list("ACGT"), 200)) for _ in range(n)],
        "expression_level": rng.normal(5.0, 1.0, size=n),
    })
    train = df.iloc[:150].copy()
    val = df.iloc[150:].copy()
    return train, val


def test_global_mean_returns_metrics(toy_data):
    train, val = toy_data
    m = baselines.global_mean(train, val, "expression_level")
    assert "R2" in m and "MSE" in m and "MAPE_percent" in m


def test_tissue_mean_produces_finite_values(toy_data):
    train, val = toy_data
    m = baselines.tissue_mean(train, val, "expression_level", "tissue")
    assert np.isfinite(m["R2"])
    assert np.isfinite(m["MSE"])


def test_run_all_returns_expected_keys(toy_data):
    train, val = toy_data
    results = baselines.run_all(train, val)
    expected = {"global_mean", "tissue_mean",
                "gc_length_tissue_ridge", "kmer4_tissue_ridge"}
    assert set(results.keys()) == expected
