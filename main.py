# https://github.com/google-research/google-research/tree/master/goemotions/data/full_dataset
# Buradkai veri setiyle metin işleme yaparak. Gelen yorumdan o yorumdaki genel duygu tutumunu tahmin eden modeli geliştirelim.
# 1. fark => Sınıflandırma => 27 farklı duygu türü
####3 csv dosyası var. birleştirip, tek csv dosyası oluştur. (yazar adı link vs duygu deeğerleri)
#teslim hafta perşembe 10 Temmuz

import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# (İsteğe Bağlı) Transformers için kütüphaneler
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# 1. Veri Yolu ve Yükleme
data_path = Path("/mnt/data/goemotions_combined.csv")
df = pd.read_csv(data_path)

# 2. Temizleme Fonksiyonu
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"[^a-zçğıöşü0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# 3. Temizlenmiş Metin Sütunu
df['text_clean'] = df['text'].apply(clean_text)

# 4. Etiketlerin Belirlenmesi (Çok Sınıflı)
emotion_cols = [c for c in df.columns if c not in ['text','author','link','text_clean']]
df['label'] = df[emotion_cols].values.argmax(axis=1)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)

# 6. Basit TF-IDF + Lojistik Regresyon Pipeline
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial"))
])
print("[Baseline] Eğitim başladı...")
pipe.fit(X_train, y_train)
print("[Baseline] Eğitim tamamlandı, değerlendirme:")
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# 7. HuggingFace Transformers ile Fine-Tuning
print("[Transformer] Veri seti HF formatına dönüştürülüyor...")
# HF Dataset formatına çevirme
hf_ds = Dataset.from_pandas(df[['text_clean','label']])
hf_ds = hf_ds.train_test_split(test_size=0.2, seed=42, stratify_by_column='label')

# Model ve tokenizer
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(emotion_cols)
)

def tokenize_fn(batch):
    return tokenizer(batch['text_clean'], truncation=True, padding='max_length', max_length=128)

hf_ds = hf_ds.map(tokenize_fn, batched=True)
hf_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])

# Eğitim ayarları
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_ds['train'],
    eval_dataset=hf_ds['test'],
)

print("[Transformer] Eğitim başladı...")
trainer.train()
print("[Transformer] Eğitim tamamlandı. Değerlendirme için çıktı dosyalarına bakın.")

# 8. Modelleri kaydetme
import joblib
# Baseline pipeline kaydetme\joblib.dump(pipe, 'tfidf_logreg_goemotions.pkl')
# Transformer modeli kaydetme
model.save_pretrained('bert_goemotions_model')
tokenizer.save_pretrained('bert_goemotions_tokenizer')

print("Modeller kaydedildi: 'tfidf_logreg_goemotions.pkl', 'bert_goemotions_model/' ve tokenizer/.")
