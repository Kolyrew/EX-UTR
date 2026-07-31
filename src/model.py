"""UtrExpressionModel: UTR-BERT encoder + tissue embedding + regression head."""
from __future__ import annotations

import torch
import torch.nn as nn


class UtrExpressionModel(nn.Module):
    """
    Predicts a scalar expression value (in log-scale) from a tokenized
    UTR5 <SEP> UTR3 sequence and a tissue identifier.

    Architecture
    ------------
      UTR-BERT (frozen for the first few epochs, then fine-tuned)
        │
        └── pooled_output  ─┐
                            │  concat  ──► Dropout ──► Linear ──► scalar
      tissue_id ──► Embed  ─┘
    """

    def __init__(
        self,
        pretrained_name: str,
        n_tissues: int,
        tissue_embed_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Lazy import: multimolecule is a heavy dependency, only needed here.
        from multimolecule.models.utrbert.modeling_utrbert import UtrBertModel

        self.encoder = UtrBertModel.from_pretrained(pretrained_name)
        hidden = self.encoder.config.hidden_size

        self.tissue_emb = nn.Embedding(n_tissues, tissue_embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden + tissue_embed_dim, 1)

    def freeze_encoder(self, freeze: bool = True) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not freeze

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tissue_id: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output                          # (B, H)
        te = self.tissue_emb(tissue_id)                     # (B, D)
        x = torch.cat([pooled, te], dim=-1)                 # (B, H+D)
        x = self.dropout(x)
        return self.regressor(x).squeeze(-1)                # (B,)
