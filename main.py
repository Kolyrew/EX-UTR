from tqdm.auto import tqdm
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    BertTokenizer,
    BertGenerationEncoder,
    BertGenerationDecoder,
    EncoderDecoderModel,
    get_linear_schedule_with_warmup
)

# 1) Параметры
DATA_PATH = r"C:\Users\lutch\PycharmProjects\EX-UTR\Data\expression_utr_summary_500.csv"
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 1
EPOCHS = 5
LR = 5e-5
MAX_INPUT_LEN = 32
MAX_OUTPUT_LEN = 200  # достаточно, чтобы уместить UTR5 + SEP + UTR3

# 2) Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3) Чтение данных
df = pd.read_csv(DATA_PATH)

class UtrDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_len, max_output_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gene = row["gene_symbol"]
        tissue = row["tissue"]
        expr = float(row["expression_level"])
        utr5 = row["UTR5_Sequence"]
        utr3 = row["UTR3_Sequence"]

        # Вход — строка "BRCA1 liver 12.34"
        input_text = f"{gene} {tissue} {expr:.2f}"
        enc = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_input_len,
            return_tensors="pt",
        )

        # Нужно "ATG... [SEP] ...TGA"
        sep = self.tokenizer.sep_token
        target_text = utr5 + " " + sep + " " + utr3
        labels = self.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_output_len,
            return_tensors="pt",
        )["input_ids"]


        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

# 4) Токенизатор и модель
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

encoder = BertGenerationEncoder.from_pretrained(
    MODEL_NAME,
    bos_token_id=tokenizer.cls_token_id,
    eos_token_id=tokenizer.sep_token_id,
)

decoder = BertGenerationDecoder.from_pretrained(
    MODEL_NAME,
    add_cross_attention=True,
    is_decoder=True,
    bos_token_id=tokenizer.cls_token_id,
    eos_token_id=tokenizer.sep_token_id,
)

model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model = model.to(device)

# 5) Датасеты и загрузчики
dataset = UtrDataset(df, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
n = len(dataset)
if n >= 2:
    train_size = max(1, int(0.9 * n))
    val_size   = n - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
else:
    # иначе используем весь датасет, валидации не будет
    train_ds = dataset
    val_ds   = []

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
if len(val_ds) > 0:
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
else:
    val_loader = None


# 6) Оптимизатор и шедулер
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# 7) Тренировочный цикл
for epoch in range(1, EPOCHS + 1):
    # TRAIN
    model.train()
    total_train_loss = 0
    train_iter = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]", leave=False)

    for batch in train_iter:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        train_iter.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch} ▶ Train loss: {avg_train_loss:.4f}")

    # VALIDATION
    if val_loader:
        model.eval()
        total_val_loss = 0
        val_tokens, val_exact, val_token_matches = 0, 0, 0

        val_iter = tqdm(val_loader, desc=f"Epoch {epoch} [VAL]", leave=False)
        with torch.no_grad():
            for batch in val_iter:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_val_loss += outputs.loss.item()

                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=MAX_OUTPUT_LEN,
                    num_beams=1,
                )

                # восстановим true_ids без -100
                true_ids = labels.clone()
                true_ids[true_ids == -100] = tokenizer.pad_token_id

                preds = gen_ids.cpu().numpy()
                trues = true_ids.cpu().numpy()

                for p, t in zip(preds, trues):
                    # истинная последовательность (без паддинга)
                    mask = (t != tokenizer.pad_token_id)
                    true_seq = t[mask]
                    true_len = len(true_seq)

                    # предсказанная (тоже без паддинга)
                    pred_list = p.tolist()
                    if tokenizer.pad_token_id in pred_list:
                        pred_list = pred_list[:pred_list.index(tokenizer.pad_token_id)]
                    pred_len = len(pred_list)

                    # exact match
                    if pred_len == true_len and pred_list == true_seq.tolist():
                        val_exact += 1

                    # token-level accuracy
                    L = min(true_len, pred_len)
                    if L > 0:
                        matches = np.array(pred_list[:L]) == np.array(true_seq[:L])
                        correct = int(matches.sum())
                        val_token_matches += correct
                    val_tokens += true_len

                val_iter.set_postfix(loss=outputs.loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        token_acc = val_token_matches / val_tokens if val_tokens else 0.0
        exact_acc = val_exact / len(val_loader.dataset)
        print(
            f"Epoch {epoch} ▶ Val loss: {avg_val_loss:.4f}  "
            f"Token-acc: {token_acc:.4f}  Exact: {exact_acc:.4f}"
        )

# 8) Сохранение чекпоинта
os.makedirs("checkpoints", exist_ok=True)
model.save_pretrained("checkpoints/bert2utr")
tokenizer.save_pretrained("checkpoints/bert2utr")

# 9) Пример инференса
print("\n=== INFERENCE ===")
model.eval()

from transformers import BatchEncoding
import torch

# Функция для одного образца
def infer_row(row):
    txt = f"{row['gene_symbol']} {row['tissue']} {row['expression_level']:.2f}"
    enc: BatchEncoding = tokenizer(
        txt,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LEN,
        return_tensors="pt"
    ).to(device)

    gen_ids = model.generate(
        enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_length=MAX_OUTPUT_LEN,
        num_beams=4,
        early_stopping=True,
    )
    pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    # Разделяем по SEP
    sep = tokenizer.sep_token
    parts = pred.split(sep)
    utr5_pred = parts[0].strip() if len(parts) > 0 else ""
    utr3_pred = parts[1].strip() if len(parts) > 1 else ""
    print("Input :", txt)
    print("UTR5  :", utr5_pred)
    print("UTR3  :", utr3_pred)
    print()

# Когда полноценный val_loader
if isinstance(val_loader, DataLoader):
    # возьмём первый индекс из val_ds
    # val_ds — это Subset, у него есть атрибут .indices
    idx = val_ds.indices[0]
    row = df.iloc[idx]
    infer_row(row)

# Когда нет валидации (val_loader == None)
elif len(df) >= 1:
    # просто инференсим по первой строке датасета
    row = df.iloc[0]
    infer_row(row)
