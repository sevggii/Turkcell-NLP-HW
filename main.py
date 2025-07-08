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
