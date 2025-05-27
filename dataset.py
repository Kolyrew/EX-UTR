import pandas as pd
import torch
from torch.utils.data import Dataset

class UtrDataset(Dataset):
    """
    Dataset for UTR prediction.
    Each sample: input_ids, attention_mask, labels with -100 for padding.
    """
    def __init__(self, df: pd.DataFrame, tokenizer, max_input_len: int, max_output_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        gene = row["gene_symbol"]
        tissue = row["tissue"]
        expr = float(row["expression_level"])
        utr5 = row["UTR5_Sequence"]
        utr3 = row["UTR3_Sequence"]

        # Input text
        input_text = f"{gene} {tissue} {expr:.2f}"
        enc = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_input_len,
            return_tensors="pt",
        )

        # Target text with SEP token
        sep_tok = self.tokenizer.sep_token
        target_text = utr5 + " " + sep_tok + " " + utr3
        labels = self.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_output_len,
            return_tensors="pt",
        )["input_ids"]

        # Replace pad token id with -100 for loss masking
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }