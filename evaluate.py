import os
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer, EncoderDecoderModel

from dataset import UtrDataset  # убедись, что dataset.py лежит рядом

# 1) Параметры — должны совпадать с тем, что ты использовал при тренировке
DATA_PATH     = r"Data/expression_utr_summary_500_2nd.csv"
CHECKPOINT    = r"checkpoints/ex-utr"              # путь к твоей сохранённой модели
BATCH_SIZE    = 8
MAX_INPUT_LEN = 32
MAX_OUTPUT_LEN= 200

# 2) Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3) Загрузка токенизатора и модели из чекпоинта
tokenizer = BertTokenizer.from_pretrained(CHECKPOINT)
model     = EncoderDecoderModel.from_pretrained(CHECKPOINT).to(device)
model.eval()

# 4) Подготовка тестового датасета
test_df      = pd.read_csv(DATA_PATH)
test_dataset = UtrDataset(test_df, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5) Оценка
total_loss     = 0.0
exact_matches  = 0
token_matches  = 0
total_tokens   = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="TEST"):
        input_ids     = batch["input_ids"].to(device)
        attention_mask= batch["attention_mask"].to(device)
        labels        = batch["labels"].to(device)

        # прямой проход + лосс
        outputs    = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels)
        total_loss+= outputs.loss.item()

        # генерация
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_OUTPUT_LEN,
            num_beams=1,
        )

        # подготовка «истины»
        true_ids = labels.clone()
        true_ids[true_ids == -100] = tokenizer.pad_token_id

        preds = gen_ids.cpu().numpy()
        trues = true_ids.cpu().numpy()

        # считаем метрики
        for p, t in zip(preds, trues):
            mask     = (t != tokenizer.pad_token_id)
            true_seq = t[mask]
            mask = (t != tokenizer.pad_token_id)
            true_seq = t[mask]
            # Обрезаем pred_seq тоже по true_seq
            pred_seq = p.tolist()
            if tokenizer.pad_token_id in pred_seq:
                pred_seq = pred_seq[:pred_seq.index(tokenizer.pad_token_id)]
            # Минимальная длина для сравнения
            L = min(len(true_seq), len(pred_seq))
            # Exact-sequence (строгий матч всей true_seq)
            if len(pred_seq) == len(true_seq) and np.array_equal(pred_seq, true_seq):
                exact_matches += 1
            # Token-level на первых L токенах
            token_matches += int((np.array(pred_seq[:L]) == true_seq[:L]).sum())
            total_tokens += len(true_seq)

# 6) Итоги
avg_loss   = total_loss / len(test_loader)
tok_acc    = token_matches / total_tokens
exact_acc  = exact_matches / len(test_dataset)

print(f"\nTest Loss:            {avg_loss:.4f}")
print(f"Token-level Accuracy: {tok_acc:.4f}")
print(f"Exact-seq Accuracy:   {exact_acc:.4f}")

print("\n=== SAMPLE PREDICTIONS ===")
n_samples = 10
for i in range(n_samples):
    row = test_df.iloc[i]
    inp_txt = f"{row['gene_symbol']} {row['tissue']} {row['expression_level']:.2f}"

    # encode
    enc = tokenizer(
        inp_txt,
        padding="max_length",
        truncation=True,
        max_length=MAX_INPUT_LEN,
        return_tensors="pt"
    ).to(device)

    # generate
    gen_ids = model.generate(
        enc["input_ids"],
        attention_mask=enc["attention_mask"],
        min_length=10,
        max_length=100,
        num_beams=4,
        early_stopping=False,
    )
    raw_ids = gen_ids[0].tolist()  # <— сохраняем сюда
    print(" Raw IDs:", raw_ids)
    print(" Tokens  :", tokenizer.convert_ids_to_tokens(raw_ids)[:20], "…")
    pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # split by SEP
    parts = pred.split(tokenizer.sep_token)
    utr5_pred = parts[0].strip() if len(parts) > 0 else ""
    utr3_pred = parts[1].strip() if len(parts) > 1 else ""

    # истинные
    true5 = row["UTR5_Sequence"]
    true3 = row["UTR3_Sequence"]

    print(f"\nSample {i + 1}:")
    print(" Input:  ", inp_txt)
    print(" True5:  ", true5[:50] + "…")  # обрезаем очень длинные
    print(" Pred5:  ", utr5_pred[:50] + "…")
    print(" True3:  ", true3[:50] + "…")
    print(" Pred3:  ", utr3_pred[:50] + "…")

