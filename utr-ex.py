from tqdm.auto import tqdm
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# 1) Импорт RNA-токенизатора и UTR-BERT
from multimolecule import RnaTokenizer
from multimolecule.models.utrbert.modeling_utrbert import UtrBertModel

# 2) PyTorch модули
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


# ───────────────── ПАРАМЕТРЫ ─────────────────
DATA_PATH     = r"C:\Users\lutch\PycharmProjects\EX-UTR\Data\expression_utr_summary_9000.csv"
MODEL_NAME    = "multimolecule/utrbert-5mer"  # или utrbert-3mer
BATCH_SIZE    = 8
EPOCHS        = 12
LR            = 2e-5
MAX_SEQ_LEN   = 400   # для UTR5+SEP+UTR3, можно «400» или «512»

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TISSUES = ["Brain","Spinal cord","Heart","Thyroid gland",
           "Lung","Liver","Pancreas","Small intestine","Colon","Kidney"]
tissue2id = {t:i for i,t in enumerate(TISSUES)}

# ───────────────── Dataset ─────────────────
class ExpressionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: RnaTokenizer, max_seq_len: int, target_log: bool = True):
        """
        df: DataFrame с колонками UTR5_Sequence, UTR3_Sequence, expression_level
        tokenizer: RnaTokenizer
        max_seq_len: максимальная длина входной токенизированной строки (UTR5 + SEP + UTR3)
        target_log: если True, то в лоссе будем предсказывать log(expr+1)
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.target_log = target_log

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        utr5 = row["UTR5_Sequence"].replace("T", "U")
        utr3 = row["UTR3_Sequence"].replace("T", "U")
        expr = float(row["expression_level"])
        if self.target_log:
            expr = np.log1p(expr)  # предсказываем лог

        # Формируем входную строку: "AUGC... [SEP] UAGG..."
        sep_token = self.tokenizer.sep_token  # строковый SEP
        text = utr5 + " " + sep_token + " " + utr3

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )
        # enc["input_ids"], enc["attention_mask"] имеют shape=(1, max_seq_len)
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        tissue = row["tissue"]
        tid = tissue2id[tissue]

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "tissue_id": torch.tensor(tid, dtype=torch.long),
            "expression":     torch.tensor(expr, dtype=torch.float),
        }

# ───────────────── Модель ─────────────────
class UtrExpressionModel(nn.Module):
    def __init__(self, model_name: str, num_tissues: int, tissue_emb_dim: int = 16):
        super().__init__()
        # 1) Загружаем UTR-BERT ядро
        self.bert = UtrBertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size   # обычно 768

        # embedding для ткани
        self.tissue_emb = nn.Embedding(num_tissues, tissue_emb_dim)

        # регрессор на hidden_size + tissue_emb_dim
        self.dropout = nn.Dropout(p=0.1)
        self.regressor = nn.Linear(hidden_size + tissue_emb_dim, 1)   # выдаёт скаляр

    def forward(self, input_ids, attention_mask, tissue_id):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output                       # (bs, hidden_size)
        te = self.tissue_emb(tissue_id)                     # (bs, tissue_emb_dim)
        x = torch.cat([pooled, te], dim=-1)                 # (bs, hidden+tissue_emb)
        x = self.dropout(x)
        return self.regressor(x).squeeze(-1)

# ───────────────── Тренировочная функция ─────────────────
def train():
    # 1) Читаем CSV
    df = pd.read_csv(DATA_PATH)

    # 2) Готовим токенизатор + датасет
    tokenizer = RnaTokenizer.from_pretrained(MODEL_NAME, nmers=5)
    dataset = ExpressionDataset(df, tokenizer, max_seq_len=MAX_SEQ_LEN, target_log=True)

    # 3) Train/val split (90/10)
    n = len(dataset)
    train_size = int(0.9 * n)
    val_size   = n - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 4) Инициализируем модель, оптимизатор, scheduler
    model = UtrExpressionModel(MODEL_NAME, num_tissues=len(TISSUES), tissue_emb_dim=16).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),  # 5%
        num_training_steps=total_steps
    )

    # 5) Лосс для регрессии
    criterion = nn.MSELoss()  # MSE на log(expr+1)

    best_val_mse = float("inf")
    best_epoch = -1

    # 6) Тренировочный цикл
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tissue_id = batch["tissue_id"].to(device)  # <- tissue ids
            target = batch["expression"].to(device)

            optimizer.zero_grad()
            preds = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          tissue_id=tissue_id)  # <- передаём tissue_id
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch} ▶ Train MSE loss: {avg_train_loss:.4f}")

        # 7) Валидация
        model.eval()
        total_val_loss = 0.0
        preds_all, trues_all = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [VAL]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                tissue_id = batch["tissue_id"].to(device)
                target = batch["expression"].to(device)

                # теперь всё будет согласовано
                preds = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tissue_id=tissue_id,
                )
                loss = criterion(preds, target)
                total_val_loss += loss.item()

                preds_all.append(preds.cpu().numpy())
                trues_all.append(target.cpu().numpy())

        # 8) Можно посчитать R²-метрику
        preds_all = np.concatenate(preds_all)
        trues_all = np.concatenate(trues_all)
        # Обратно восстанавливаем исходную шкалу: expm1
        preds_orig = np.expm1(preds_all)
        trues_orig = np.expm1(trues_all)
        r2 = 1 - ((preds_orig - trues_orig) ** 2).sum() / ((trues_orig - trues_orig.mean()) ** 2).sum()
        print(f"Epoch {epoch} ▶ Val R² (on expr scale): {r2:.4f}\n")

    # 9) Сохраняем чекпоинт
    os.makedirs("checkpoints_expression", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints_expression/utrbert_expr.pth")
    tokenizer.save_pretrained("checkpoints_expression/")
    print("✓ Модель и токенизатор сохранены в ‘checkpoints_expression’")

if __name__ == "__main__":
    train()
