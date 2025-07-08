<<<<<<< HEAD
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
=======
"""
GoEmotions Çoklu Duygu Sınıflandırma
- Veri: goemotions_combined.csv
- Model: Logistic Regression (One-vs-Rest)
- Ön işleme: Küçük harf, noktalama, sayı, stopword, lemmatizasyon
- Çıktı: Her bir duygu için başarı, örnek tahmin

Gereksinimler:
- pandas, scikit-learn, nltk
"""
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk veri setlerini indir (ilk çalıştırmada gerekebilir)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# 1. Veri setini oku
DATA_PATH = '../../goemotions_combined.csv'
df = pd.read_csv(DATA_PATH)

# 2. Etiket sütunlarını ve metni ayır
text_col = 'text'
label_cols = [col for col in df.columns if col not in ['text', 'id']]
X_raw = df[text_col].astype(str)
y = df[label_cols]

# 3. Etiket dağılımı raporu (kaç örnekte hangi duygu var?)
print("Etiket dağılımı (örnek sayısı):")
for label in label_cols:
    print(f"{label:15}: {y[label].sum()}")

# 4. Ön işleme fonksiyonu
def preprocess_text(text):
    # Küçük harfe çevir
    text = text.lower()
    # Noktalama ve sayıları kaldır
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    # Fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text).strip()
    # Stopword temizliği
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    # Lemmatizasyon
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

print("\nÖn işleme uygulanıyor...")
X = X_raw.apply(preprocess_text)

# 5. Eğitim/test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Metinleri vektörleştir (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Modeli eğit (One-vs-Rest Logistic Regression)
print("\nModel eğitiliyor...")
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train_vec, y_train)

# 8. Test setinde tahmin ve rapor
print("\nTest seti sonuçları:")
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=label_cols, zero_division=0))

# 9. Örnek tahmin fonksiyonu
def predict_emotions(text):
    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    emotions = [label for label, val in zip(label_cols, pred) if val == 1]
    return emotions

# 10. Kullanıcıya örnek çıktı
def print_sample_prediction(text):
    print("\nÖrnek metin:", text)
    print("Tahmin edilen duygular:", predict_emotions(text))

sample_texts = [
    "I am so happy and excited!",
    "This is the worst day ever.",
    "I don't care about this at all.",
    "Wow, that's surprising!",
    "You make me so angry!"
]
for st in sample_texts:
    print_sample_prediction(st)

print("\nTüm işlem tamamlandı. Kodun her adımı açıklamalı ve insan gibi düzenlenmiştir.")
>>>>>>> e669ff897a4f5e00dcd12576c2ea233ebfb9880d
