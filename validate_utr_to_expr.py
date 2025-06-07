# validate_utr_to_expr_with_tissue.py
import random
import math
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from multimolecule import RnaTokenizer
from multimolecule.models.utrbert.modeling_utrbert import UtrBertModel
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# ───────────── ПАРАМЕТРЫ ─────────────
DATA_PATH      = r"C:\Users\lutch\PycharmProjects\EX-UTR\Data\expression_utr_summary_1200.csv"
CHECKPOINT_DIR = "checkpoints_expression"   # папка с utrbert_expr.pth + tokenizer
MODEL_NAME     = "multimolecule/utrbert-5mer"
MAX_SEQ_LEN    = 400

# тisssue mapping
TISSUES = ["Brain","Spinal cord","Heart","Thyroid gland",
           "Lung","Liver","Pancreas","Small intestine","Colon","Kidney"]
tissue2id = {t:i for i,t in enumerate(TISSUES)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────── Модель с учётом ткани ─────────────
class UtrExpressionModelWithTissue(nn.Module):
    def __init__(self, model_name: str, num_tissues: int, tissue_emb_dim: int = 16):
        super().__init__()
        self.bert = UtrBertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.tissue_emb = nn.Embedding(num_tissues, tissue_emb_dim)
        self.dropout = nn.Dropout(p=0.1)
        # регрессор на concat([CLS], tissue_emb)
        self.regressor = nn.Linear(hidden_size + tissue_emb_dim, 1)

    def forward(self, input_ids, attention_mask, tissue_id):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = outputs.pooler_output                           # (bs, hidden_size)
        te = self.tissue_emb(tissue_id)                         # (bs, tissue_emb_dim)
        x = torch.cat([pooled, te], dim=-1)                     # (bs, hidden + tissue_emb)
        x = self.dropout(x)
        return self.regressor(x).squeeze(-1)                     # (bs,)

# ───────────── ЗАГРУЗКА И SPLIT ─────────────
df = pd.read_csv(DATA_PATH)
train_idx, val_idx = train_test_split(df.index, test_size=0.1, random_state=42)
val_df = df.loc[val_idx].reset_index(drop=True)

# ───────────── СОЗДАЁМ МОДЕЛЬ И ЗАГРУЗКА ВЕСОВ ─────────────
model = UtrExpressionModelWithTissue(
    model_name=MODEL_NAME,
    num_tissues=len(TISSUES),
    tissue_emb_dim=16
).to(device)

state_dict = torch.load(f"{CHECKPOINT_DIR}/utrbert_expr.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ───────────── ТОКЕНИЗАТОР ─────────────
tokenizer = RnaTokenizer.from_pretrained(CHECKPOINT_DIR)

# ───────────── ФУНКЦИЯ ПРЕДСКАЗАНИЯ ─────────────
def predict_expression(utr5: str, utr3: str, tissue: str):
    rna5 = utr5.replace("T","U")
    rna3 = utr3.replace("T","U")
    sep = tokenizer.sep_token
    text = rna5 + " " + sep + " " + rna3
    enc = tokenizer(text,
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                    return_tensors="pt").to(device)
    tid = torch.tensor([tissue2id[tissue]], dtype=torch.long, device=device)
    with torch.no_grad():
        y_log = model(input_ids=enc["input_ids"],
                      attention_mask=enc["attention_mask"],
                      tissue_id=tid)
    return y_log.item()

# ───────────── СРАВНИТЕЛЬНЫЕ ПРЕДСКАЗАНИЯ ─────────────
random.seed(123)
sample_idxs = random.sample(list(range(len(val_df))), k=10)

print(" idx |     Gene      | Tissue       |  True expr  |  Pred expr ")
print("-----+---------------+--------------+-------------+-------------")
for i in sample_idxs:
    row = val_df.iloc[i]
    gene = row["gene_symbol"]
    tissue = row["tissue"]
    utr5, utr3 = row["UTR5_Sequence"], row["UTR3_Sequence"]
    true_expr = float(row["expression_level"])
    pred_log = predict_expression(utr5, utr3, tissue)
    pred_expr = float(np.expm1(pred_log))
    print(f"{i:4d} | {gene:13s} | {tissue:12s} | {true_expr:11.4f} | {pred_expr:11.4f}")

# ───────────── ПОЛНАЯ ВАЛИДАЦИЯ ─────────────
mse_log, mse_orig = 0.0, 0.0
r2_num_log, r2_den_log = 0.0, 0.0
r2_num_orig, r2_den_orig = 0.0, 0.0
preds_orig = []
trues_orig = []
tissues_list = []
mape_acc = []


for row in tqdm(val_df.itertuples(), total=len(val_df), desc="Full‐validation"):
    true_expr = float(row.expression_level)
    true_log = math.log1p(true_expr)
    pred_log = predict_expression(row.UTR5_Sequence, row.UTR3_Sequence, row.tissue)
    pred_expr = math.expm1(pred_log)

    preds_orig.append(pred_expr)
    trues_orig.append(true_expr)
    tissues_list.append(row.tissue)
    mape_acc.append(abs(pred_expr - true_expr) / (true_expr + 1e-8))

    # accumulate for MSE
    mse_log  += (pred_log  - true_log)**2
    mse_orig += (pred_expr - true_expr)**2

    # accumulate for R²
    r2_num_log  += (pred_log  - true_log)**2
    r2_den_log  += (true_log  - np.log1p(val_df["expression_level"]).mean())**2

    r2_num_orig += (pred_expr - true_expr)**2
    r2_den_orig += (true_expr - val_df["expression_level"].mean())**2

    mape_acc.append(abs(pred_expr - true_expr) / (true_expr + 1e-8))

mse_log  /= len(val_df)
mse_orig /= len(val_df)
r2_log   = 1 - r2_num_log  / r2_den_log
r2_orig  = 1 - r2_num_orig / r2_den_orig
mape_pct = 100 * np.mean(mape_acc)

print(f"\nПолная валидация на {len(val_df)} примерах:")
print(f"  • MSE (log(expr+1)) : {mse_log:.4f}")
print(f"  • MSE (expr)       : {mse_orig:.4f}")
print(f"  • R² (log)         : {r2_log:.4f}")
print(f"  • R² (orig)        : {r2_orig:.4f}")
print(f"  • MAPE (%) (expr)  : {mape_pct:.2f}%")

df_res = pd.DataFrame({
    "tissue": tissues_list,
    "true_expr": trues_orig,
    "pred_expr": preds_orig,
})

print("\n=== Потканевые метрики ===")
for tissue, group in df_res.groupby("tissue"):
    y_true = group["true_expr"].values
    y_pred = group["pred_expr"].values
    mse   = mean_squared_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    mape  = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"{tissue:15s}  MSE={mse:.3f}  R²={r2:.3f}  MAPE={mape:.1f}%")

