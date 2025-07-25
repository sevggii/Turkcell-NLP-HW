##Merhabalar bu ödev Sevgi Targay ve Büşranur Çevik tarafından hazırlanmıştır, 

import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer
from transformers import Seq2SeqTrainingArguments as TrainingArguments

from datasets import Dataset
import evaluate
import torch

# ----------------------
# 1. Veri Yükleme
# ----------------------
df = pd.read_csv("cnn_dailymail/validation.csv") # veya test.csv

df.rename(columns={"highlights": "summary"}, inplace=True)

df = df[["article", "summary"]]  # Sadece gerekli sütunlar kalsın



# ----------------------
# 2. Ön İşleme Fonksiyonu
# ----------------------
def clean_text(text):
    if isinstance(text, str):
        return text.lower().strip()
    return ""

df['article'] = df['article'].apply(clean_text)
df['summary'] = df['summary'].apply(clean_text)

# ----------------------
# 3. Tokenizer ve Uzunluk Ayarları
# ----------------------
tokenizer = T5Tokenizer.from_pretrained("t5-small")
max_input = 256
max_target = 64

# ----------------------
# 4. Tokenizasyon
# ----------------------
def tokenize_function(example):
    inputs = tokenizer(
        example["article"],
        max_length=max_input,
        padding="max_length",
        truncation=True
    )
    targets = tokenizer(
        example["summary"],
        max_length=max_target,
        padding="max_length",
        truncation=True
    )
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": targets["input_ids"]
    }




raw_dataset = Dataset.from_pandas(df[["article", "summary"]])
tokenized_dataset = raw_dataset.map(tokenize_function, remove_columns=["article", "summary"])


# ----------------------
# 5. Model Kurulumu
# ----------------------
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# ----------------------
# 6. Eğitim Ayarları
# ----------------------
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,
    remove_unused_columns=False,
)

# ----------------------
# 7. Trainer ve Eğitim
# ----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# ----------------------
# 8. Özet Üretme
# ----------------------
def generate_summary(example):
    input_ids = tokenizer.encode(example['article'], return_tensors='pt', max_length=max_input, truncation=True)
    output_ids = model.generate(input_ids, max_length=max_target, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

df['generated_summary'] = df.apply(generate_summary, axis=1)

# ----------------------
# 9. ROUGE Değerlendirmesi
# ----------------------
rouge = evaluate.load("rouge")
results = rouge.compute(predictions=df["generated_summary"].tolist(), references=df["summary"].tolist())
print("ROUGE sonuçları:", results)

# ----------------------
# 10. Örnek Çıktılar
# ----------------------
for i in range(5):
    print(f"\n--- Örnek {i+1} ---")
    print("[Girdi]", df['article'][i])
    print("[Beklenen]", df['summary'][i])
    print("[Üretilen]", df['generated_summary'][i])
