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
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import shutil

# nltk veri setlerini indir (ilk çalıştırmada gerekebilir)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# Otomatik kopyalama: Eğer CSV yoksa masaüstünden veya kullanıcıdan alınan yoldan kopyala
DATA_PATH = 'goemotions_combined.csv'
DESKTOP_PATH = os.path.expanduser("~/Desktop/goemotions_combined.csv")
if not os.path.exists(DATA_PATH):
    if os.path.exists(DESKTOP_PATH):
        try:
            shutil.copy(DESKTOP_PATH, DATA_PATH)
            print("CSV dosyası masaüstünden otomatik olarak kopyalandı.")
        except Exception as e:
            print("Kopyalama sırasında hata oluştu:", e)
            exit()
    else:
        print('HATA: goemotions_combined.csv dosyası bu dizinde ve masaüstünde bulunamadı!')
        print('Şu anki dizin:', os.getcwd())
        # Kullanıcıdan dosya yolu iste
        user_path = input('Lütfen goemotions_combined.csv dosyasının tam yolunu girin: ')
        if os.path.exists(user_path):
            try:
                shutil.copy(user_path, DATA_PATH)
                print("CSV dosyası belirtilen konumdan kopyalandı.")
            except Exception as e:
                print("Kopyalama sırasında hata oluştu:", e)
                exit()
        else:
            print('Girilen dosya yolu bulunamadı! Program sonlandırılıyor.')
            exit()

# 1. Veri yükle
print("Veri yükleniyor...")
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

# 4. Gelişmiş ön işleme fonksiyonu
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

print("\nÖn işleme uygulanıyor...")
X = X_raw.apply(preprocess_text)

# 5. Eğitim/test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Pipeline ile model
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000)))
])

# 7. Modeli eğit
print("\nModel eğitiliyor...")
pipe.fit(X_train, y_train)

# 8. Test setinde tahmin ve rapor
print("\nTest seti sonuçları:")
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_cols, zero_division=0))

# 9. Örnek tahmin fonksiyonu

def predict_emotions(text):
    clean = preprocess_text(text)
    pred = pipe.predict([clean])[0]
    emotions = [label for label, val in zip(label_cols, pred) if val == 1]
    return emotions

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

