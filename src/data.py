"""PyTorch Dataset for UTR sequences and tissue-specific expression."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TISSUES = [
    "Brain", "Spinal cord", "Heart", "Thyroid gland", "Lung",
    "Liver", "Pancreas", "Small intestine", "Colon", "Kidney",
]
TISSUE2ID = {t: i for i, t in enumerate(TISSUES)}


class ExpressionDataset(Dataset):
    """
    Yields a dict:
      - input_ids       : LongTensor  (max_seq_len,)
      - attention_mask  : LongTensor  (max_seq_len,)
      - tissue_id       : LongTensor  scalar
      - expression      : FloatTensor scalar  (in log(1+y) if log1p_target=True)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        utr5_column: str = "UTR5_Sequence",
        utr3_column: str = "UTR3_Sequence",
        tissue_column: str = "tissue",
        target_column: str = "expression_level",
        max_seq_len: int = 400,
        log1p_target: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.utr5_col = utr5_column
        self.utr3_col = utr3_column
        self.tissue_col = tissue_column
        self.target_col = target_column
        self.max_seq_len = max_seq_len
        self.log1p_target = log1p_target

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        utr5 = str(row[self.utr5_col]).replace("T", "U")
        utr3 = str(row[self.utr3_col]).replace("T", "U")
        expr = float(row[self.target_col])
        if self.log1p_target:
            expr = float(np.log1p(expr))

        # Build the tokenizer input: UTR5 <SEP> UTR3
        sep = self.tokenizer.sep_token
        text = f"{utr5} {sep} {utr3}"

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "tissue_id": torch.tensor(TISSUE2ID[row[self.tissue_col]], dtype=torch.long),
            "expression": torch.tensor(expr, dtype=torch.float),
        }
