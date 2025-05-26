# main.py

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
DATA_PATH = r"C:\Users\lutch\PycharmProjects\EX-UTR\Data\expression_utr_summary_1.csv"
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 2
EPOCHS = 5
LR = 5e-5
MAX_INPUT_LEN = 32
MAX_OUTPUT_LEN = 200  # достаточно, чтобы уместить UTR5 + SEP + UTR3

# 2) Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3) Читаем данные
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

        # Цель — "ATG... [SEP] ...TGA"
        sep = self.tokenizer.sep_token
        target_text = utr5 + " " + sep + " " + utr3
        labels = self.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_output_len,
            return_tensors="pt",
        )["input_ids"]

        # Маска для лосса: паддинги = -100
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
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

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
    model.train()
    total_loss = 0
    for batch in train_loader:
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

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} — Train loss: {avg_train_loss:.4f}")

    # Валидация
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
            )
            total_val_loss += outputs.loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch} —  Val loss: {avg_val_loss:.4f}")

# 8) Сохранение чекпоинта
os.makedirs("checkpoints", exist_ok=True)
model.save_pretrained("checkpoints/bert2utr")
tokenizer.save_pretrained("checkpoints/bert2utr")

# 9) Пример инференса на первой строке валидации
model.eval()
row = df.iloc[val_ds.indices[0]]
txt = f"{row['gene_symbol']} {row['tissue']} {row['expression_level']:.2f}"
enc = tokenizer(txt, return_tensors="pt").to(device)
gen_ids = model.generate(
    enc["input_ids"],
    attention_mask=enc["attention_mask"],
    max_length=MAX_OUTPUT_LEN,
    num_beams=4,
    early_stopping=True,
)
pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
utr5_pred, utr3_pred = pred.split(tokenizer.sep_token)
print("Input :", txt)
print("Output:", utr5_pred, utr3_pred)
