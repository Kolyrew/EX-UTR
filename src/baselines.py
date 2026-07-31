"""Baseline models: how well can we do *without* deep learning?

Purpose:
    Establish a floor before adding neural machinery. If a Ridge regression on
    k-mer counts + tissue one-hot beats the neural model on the honest split,
    then the neural model isn't learning anything the simple features can't
    already provide.

Implemented baselines:
    1. Global mean            — always predicts the training set mean.
    2. Tissue mean            — predicts the mean per tissue from train.
    3. GC + length + tissue   — Ridge regression on simple hand-crafted features.
    4. k-mer + tissue         — Ridge on k-mer count vectors (k=4 by default).
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "MAPE_percent": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def _gc(seq: str) -> float:
    if not isinstance(seq, str) or len(seq) == 0:
        return 0.5
    return (seq.count("G") + seq.count("C")) / len(seq)


def _kmer_vector(seq: str, k: int, all_kmers: list[str]) -> np.ndarray:
    if not isinstance(seq, str):
        return np.zeros(len(all_kmers))
    counts: dict[str, int] = {}
    for i in range(len(seq) - k + 1):
        km = seq[i : i + k]
        if all(c in "ACGT" for c in km):
            counts[km] = counts.get(km, 0) + 1
    total = sum(counts.values()) or 1
    return np.array([counts.get(km, 0) / total for km in all_kmers])


# ---------------------------------------------------------------------------
def global_mean(train: pd.DataFrame, val: pd.DataFrame, target_col: str) -> dict:
    mean = train[target_col].mean()
    pred = np.full(len(val), mean)
    return _metrics(val[target_col].values, pred)


def tissue_mean(
    train: pd.DataFrame, val: pd.DataFrame, target_col: str, tissue_col: str
) -> dict:
    means = train.groupby(tissue_col)[target_col].mean()
    pred = val[tissue_col].map(means).values
    fallback = train[target_col].mean()
    pred = np.where(pd.isna(pred), fallback, pred)
    return _metrics(val[target_col].values, pred)


def gc_length_tissue_ridge(
    train: pd.DataFrame, val: pd.DataFrame,
    target_col: str, tissue_col: str, utr5_col: str, utr3_col: str,
    seed: int = 42,
) -> dict:
    def features(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "gc5": df[utr5_col].apply(_gc),
            "gc3": df[utr3_col].apply(_gc),
            "len5": df[utr5_col].str.len(),
            "len3": df[utr3_col].str.len(),
        })

    X_tr_num = features(train).values
    X_va_num = features(val).values

    dummies_tr = pd.get_dummies(train[tissue_col], prefix="t")
    dummies_va = pd.get_dummies(val[tissue_col], prefix="t").reindex(
        columns=dummies_tr.columns, fill_value=0
    )

    X_tr = np.concatenate([X_tr_num, dummies_tr.values], axis=1)
    X_va = np.concatenate([X_va_num, dummies_va.values], axis=1)

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_va = sc.transform(X_va)

    m = Ridge(alpha=1.0, random_state=seed)
    m.fit(X_tr, train[target_col].values)
    return _metrics(val[target_col].values, m.predict(X_va))


def kmer_tissue_ridge(
    train: pd.DataFrame, val: pd.DataFrame,
    target_col: str, tissue_col: str, utr5_col: str, utr3_col: str,
    k: int = 4, seed: int = 42,
) -> dict:
    all_kmers = ["".join(p) for p in product("ACGT", repeat=k)]

    def kmer_matrix(seqs: pd.Series) -> np.ndarray:
        return np.stack([_kmer_vector(s, k, all_kmers) for s in seqs])

    X5_tr = kmer_matrix(train[utr5_col])
    X3_tr = kmer_matrix(train[utr3_col])
    X5_va = kmer_matrix(val[utr5_col])
    X3_va = kmer_matrix(val[utr3_col])

    dummies_tr = pd.get_dummies(train[tissue_col], prefix="t")
    dummies_va = pd.get_dummies(val[tissue_col], prefix="t").reindex(
        columns=dummies_tr.columns, fill_value=0
    )

    X_tr = np.concatenate([X5_tr, X3_tr, dummies_tr.values], axis=1)
    X_va = np.concatenate([X5_va, X3_va, dummies_va.values], axis=1)

    m = Ridge(alpha=1.0, random_state=seed)
    m.fit(X_tr, train[target_col].values)
    return _metrics(val[target_col].values, m.predict(X_va))


def run_all(
    train: pd.DataFrame, val: pd.DataFrame,
    target_col: str = "expression_level",
    tissue_col: str = "tissue",
    utr5_col: str = "UTR5_Sequence",
    utr3_col: str = "UTR3_Sequence",
    seed: int = 42,
) -> dict[str, dict]:
    """Run all baselines and return their metrics keyed by name."""
    return {
        "global_mean":
            global_mean(train, val, target_col),
        "tissue_mean":
            tissue_mean(train, val, target_col, tissue_col),
        "gc_length_tissue_ridge":
            gc_length_tissue_ridge(train, val, target_col, tissue_col,
                                   utr5_col, utr3_col, seed=seed),
        "kmer4_tissue_ridge":
            kmer_tissue_ridge(train, val, target_col, tissue_col,
                              utr5_col, utr3_col, k=4, seed=seed),
    }
