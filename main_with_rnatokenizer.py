from tqdm.auto import tqdm
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right

# -----------------------------------------------------
#  Импорт RNA-токенизатора
from multimolecule import RnaTokenizer
# Импорт модели UTR-BERT (ядро)
from multimolecule.models.utrbert.modeling_utrbert import UtrBertModel

from transformers import (
    BertGenerationEncoder,
    BertGenerationDecoder,
    EncoderDecoderModel,
    get_linear_schedule_with_warmup
)

# 1) Параметры
DATA_PATH      = r"C:\Users\lutch\PycharmProjects\EX-UTR\Data\expression_utr_summary_500.csv"
MODEL_NAME     = "multimolecule/utrbert-5mer"
BATCH_SIZE     = 8
EPOCHS         = 10
LR             = 2e-5
MAX_INPUT_LEN  = 32
MAX_OUTPUT_LEN = 400

# 2) Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3) Чтение данных
df = pd.read_csv(DATA_PATH)

# Test 3 point
# Подсчет длины UTR5 и UTR3 в нуклеотидах
df["len5"] = df["UTR5_Sequence"].str.len()
df["len3"] = df["UTR3_Sequence"].str.len()

print("=== Статистика длин UTR5 (в нуклеотидах) ===")
print(df["len5"].describe())
print("\n=== Статистика длин UTR3 (в нуклеотидах) ===")
print(df["len3"].describe())

# Доля случаев, когда суммарная длина > 1000 (примерно 200 токенов)
total_len = df["len5"] + df["len3"]
count_over = (total_len > 1000).sum()
pct_over = 100 * count_over / len(df)
print(f"\nДоля записей с (len5+len3) > 1000: {pct_over:.2f}% ({count_over} из {len(df)})")


class UtrDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_len, max_output_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        gene  = row["gene_symbol"]
        tissue= row["tissue"]
        expr  = float(row["expression_level"])
        utr5  = row["UTR5_Sequence"]
        utr3  = row["UTR3_Sequence"]

        # Конвертация DNA->RNA: T->U
        utr5 = utr5.replace("T", "U")
        utr3 = utr3.replace("T", "U")

        # 1) токенизируем “вход”
        input_text = f"{gene} {tissue} {expr:.2f}"
        enc = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_input_len,
            return_tensors="pt",
        )

        # 2) Формируем “сырую” целевую часть без CLS/SEP
        sep = self.tokenizer.sep_token  # строковый SEP, но нам нужен ID
        raw_text = utr5 + " " + sep + " " + utr3

        # Токенизируем без специальных токенов, оставляя место для CLS и SEP
        raw_ids = self.tokenizer(
            raw_text,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_output_len - 2,
            return_tensors="pt",
        )["input_ids"].squeeze(0)  # shape = (max_output_len - 2,)

        # 3) Ручная сборка labels: [CLS] + raw_ids + [SEP] + pad...
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        labels = torch.full(
            (self.max_output_len,),
            fill_value=pad_id,
            dtype=torch.long
        )
        labels[0] = cls_id
        # вставляем “сырые” токены начиная с индекса 1
        valid_raw_len = (raw_ids != pad_id).sum().item()
        labels[1 : 1 + valid_raw_len] = raw_ids[:valid_raw_len]
        # вставляем SEP сразу после “сырых” (если остается место)
        if 1 + valid_raw_len < self.max_output_len:
            labels[1 + valid_raw_len] = sep_id

        # 4) Помечаем паддинги как -100
        labels[labels == pad_id] = -100

        return {
            "input_ids":     enc["input_ids"].squeeze(0),
            "attention_mask":enc["attention_mask"].squeeze(0),
            "labels":         labels,
        }


# -----------------------------------------------------
# 4) Токенизатор и модель

# 4.1) Инициализация RnaTokenizer (5-mer токенизация)
tokenizer = RnaTokenizer.from_pretrained(MODEL_NAME, nmers=5)

# 4.2) Загрузка UtrBertModel (ядро BERT, предобученное на UTR)
from multimolecule.models.utrbert.modeling_utrbert import UtrBertModel
utr_model = UtrBertModel.from_pretrained(MODEL_NAME)

# 4.3) Патчинг конфигурации: копируем dropout-поля (имена могут отличаться)
cfg = utr_model.config
if not hasattr(cfg, "hidden_dropout_prob") and hasattr(cfg, "hidden_dropout"):
    cfg.hidden_dropout_prob = cfg.hidden_dropout
if not hasattr(cfg, "attention_probs_dropout_prob") and hasattr(cfg, "attention_dropout"):
    cfg.attention_probs_dropout_prob = cfg.attention_dropout

# 4.4) Создание Encoder: оборачиваем BERT-ядро (base_model) в BertGenerationEncoder
encoder = BertGenerationEncoder(cfg)
encoder.bert = utr_model.base_model

# ───────────── ОТКЛЮЧАЕМ CACHE (пастовые ключи) ─────────────
cfg.use_cache = False

# 4.5) Подготовка конфига для Decoder
cfg.is_decoder = True
cfg.add_cross_attention = True

# 4.6) Создание Decoder на основе этого же BERT-ядра
decoder = BertGenerationDecoder(cfg)
decoder.bert = utr_model.base_model

# 4.7) Сборка Seq2Seq-модели и задание специальных токенов
from transformers import EncoderDecoderModel
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id             = tokenizer.sep_token_id
model.config.pad_token_id             = tokenizer.pad_token_id

model = model.to(device)

# -----------------------------------------------------
# 5) Датасеты и загрузчики
dataset = UtrDataset(df, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
n = len(dataset)
if n >= 2:
    train_size = max(1, int(0.9 * n))
    val_size   = n - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
else:
    train_ds = dataset
    val_ds   = []

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE) if val_ds else None

# -----------------------------------------------------
# 6) Оптимизатор и шедулер
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# -----------------------------------------------------
# 7) Тренировочный цикл
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0
    train_iter = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]", leave=False)

    for batch_idx, batch in enumerate(train_iter):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)  # это метки с -100 для паддинга

        # ───────────── DEBUG BLOCK ─────────────
        if epoch == 1 and batch_idx == 0:
            # проверка как выглядят первые 20 токенов labels
            print("=== DEBUG: оригинальные labels (20 токенов) ===")
            print(labels[0, :20].cpu().numpy())

            # Для вычисления decoder_input_ids заменяем -100 на pad_token_id:
            labels_for_shift = labels.clone()
            labels_for_shift[labels_for_shift == -100] = model.config.pad_token_id

            # Сдвигаем labels вправо:
            decoder_input_ids = shift_tokens_right(
                labels_for_shift,
                pad_token_id=model.config.pad_token_id,
                decoder_start_token_id=model.config.decoder_start_token_id
            )
            print("=== DEBUG: decoder_input_ids (20 токенов) после shift_tokens_right ===")
            print(decoder_input_ids[0, :20].cpu().numpy())
            print("==========================================")

        # ───────────── конец DEBUG BLOCK ─────────────

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
        total_val_loss   = 0
        val_tokens       = 0
        val_exact        = 0
        val_token_matches= 0

        val_iter = tqdm(val_loader, desc=f"Epoch {epoch} [VAL]", leave=False)
        with torch.no_grad():
            for batch in val_iter:
                input_ids     = batch["input_ids"].to(device)
                attention_mask= batch["attention_mask"].to(device)
                labels        = batch["labels"].to(device)

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

                # Восстановим true_ids без -100
                true_ids = labels.clone()
                true_ids[true_ids == tokenizer.pad_token_id] = tokenizer.pad_token_id

                preds = gen_ids.cpu().numpy()
                trues = true_ids.cpu().numpy()

                for p, t in zip(preds, trues):
                    mask     = (t != tokenizer.pad_token_id)
                    true_seq = t[mask]
                    true_len = len(true_seq)

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
        token_acc    = val_token_matches / val_tokens if val_tokens else 0.0
        exact_acc    = val_exact / len(val_loader.dataset)
        print(
            f"Epoch {epoch} ▶ Val loss: {avg_val_loss:.4f}  "
            f"Token-acc: {token_acc:.4f}  Exact: {exact_acc:.4f}"
        )


# 8) Сохранение чекпоинта
os.makedirs("checkpoints", exist_ok=True)
model.save_pretrained("checkpoints/utrbert2utr", safe_serialization=False)
tokenizer.save_pretrained("checkpoints/utrbert2utr")


# 9) Пример инференса
print("\n=== INFERENCE ===")
model.eval()

from transformers import BatchEncoding

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
    pred_rna = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    pred_dna = pred_rna.replace("U", "T")  # возврат DNA-формата

    sep = tokenizer.sep_token
    parts = pred_dna.split(sep)
    utr5_pred = parts[0].strip() if len(parts) > 0 else ""
    utr3_pred = parts[1].strip() if len(parts) > 1 else ""
    print("Input :", txt)
    print("UTR5  :", utr5_pred)
    print("UTR3  :", utr3_pred)
    print()

if val_loader:
    idx = val_ds.indices[0]
    row = df.iloc[idx]
    infer_row(row)
elif len(df) >= 1:
    row = df.iloc[0]
    infer_row(row)