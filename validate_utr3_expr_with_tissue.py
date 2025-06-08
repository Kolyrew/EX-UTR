
import random
import math
import torch
import pandas as pd
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from tqdm.auto import tqdm

from multimolecule import RnaTokenizer
from multimolecule.models.utrbert.modeling_utrbert import UtrBertModel

# ───────────────── PARAMETERS ─────────────────
DATA_PATH      = r"C:\Users\lutch\PycharmProjects\EX-UTR\Data\expression_utr_summary_1200.csv"
CHECKPOINT_DIR = "checkpoints_utr3_expr"   # папка с utr3_expr.pth + токенизатором
MODEL_NAME     = "multimolecule/utrbert-3mer"
MAX_SEQ_LEN    = 200

TISSUES = [
    "Brain","Spinal cord","Heart","Thyroid gland",
    "Lung","Liver","Pancreas","Small intestine","Colon","Kidney"
]
tissue2id = {t:i for i,t in enumerate(TISSUES)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────── Model Definition ─────────────
class Utr3ExpressionModelWithTissue(nn.Module):
    def __init__(self, model_name: str, num_tissues: int, tissue_emb_dim: int = 16):
        super().__init__()
        self.bert = UtrBertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.tissue_emb = nn.Embedding(num_tissues, tissue_emb_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.regressor = nn.Linear(hidden_size + tissue_emb_dim, 1)

    def forward(self, input_ids, attention_mask, tissue_id):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled = outputs.pooler_output               # CLS embedding
        te = self.tissue_emb(tissue_id)             # tissue embedding
        x = torch.cat([pooled, te], dim=-1)
        x = self.dropout(x)
        return self.regressor(x).squeeze(-1)

# ───────────── Load data & split ─────────────
df = pd.read_csv(DATA_PATH)
train_idx, val_idx = train_test_split(df.index, test_size=0.1, random_state=42)
val_df = df.loc[val_idx].reset_index(drop=True)

# ───────────── Instantiate model & load weights ─────────────
model = Utr3ExpressionModelWithTissue(
    model_name=MODEL_NAME,
    num_tissues=len(TISSUES),
    tissue_emb_dim=16
).to(device)

state_dict = torch.load(f"{CHECKPOINT_DIR}/utr3_expr.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ───────────── Load tokenizer ─────────────
tokenizer = RnaTokenizer.from_pretrained(CHECKPOINT_DIR)

# ───────────── Prediction helper ─────────────
def predict(utr3: str, tissue: str) -> float:
    rna3 = utr3.replace("T", "U")
    enc = tokenizer(
        rna3,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt"
    ).to(device)
    tid = torch.tensor([tissue2id[tissue]], dtype=torch.long, device=device)
    with torch.no_grad():
        y_log = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            tissue_id=tid
        )
    return y_log.item()

# ───────────── Sample predictions ─────────────
print(" idx |     Gene      | Tissue       |  True expr  |  Pred expr ")
print("-----+---------------+--------------+-------------+-------------")
random.seed(123)
for i in random.sample(range(len(val_df)), 10):
    row = val_df.iloc[i]
    pred_log  = predict(row["UTR3_Sequence"], row["tissue"])
    pred_expr = np.expm1(pred_log)
    print(f"{i:4d} | {row['gene_symbol']:13s} | {row['tissue']:12s} | "
          f"{row['expression_level']:11.4f} | {pred_expr:11.4f}")

# ───────────── Full validation ─────────────
preds, trues, tissues = [], [], []
for row in tqdm(val_df.itertuples(), total=len(val_df), desc="Full‐val"):
    true_expr = float(row.expression_level)
    pred_log  = predict(row.UTR3_Sequence, row.tissue)
    preds.append(np.expm1(pred_log))
    trues.append(true_expr)
    tissues.append(row.tissue)

preds = np.array(preds)
trues = np.array(trues)

mse    = mean_squared_error(trues, preds)
r2     = r2_score(trues, preds)
mape   = mean_absolute_percentage_error(trues, preds) * 100

print(f"\nFull validation on {len(val_df)} samples:")
print(f"  • MSE : {mse:.4f}")
print(f"  • R²  : {r2:.4f}")
print(f"  • MAPE: {mape:.2f}%\n")

# ───────────── Per‐tissue metrics ─────────────
res_df = pd.DataFrame({
    "tissue": tissues,
    "true":   trues,
    "pred":   preds
})
print("Per‐tissue metrics:")
for tissue, grp in res_df.groupby("tissue"):
    y_t, y_p = grp["true"].values, grp["pred"].values
    print(f"{tissue:15s}  "
          f"MSE={mean_squared_error(y_t,y_p):.3f}  "
          f"R²={r2_score(y_t,y_p):.3f}  "
          f"MAPE={mean_absolute_percentage_error(y_t,y_p)*100:4.1f}%")
