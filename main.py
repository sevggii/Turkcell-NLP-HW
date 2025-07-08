import re
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model    import LogisticRegression
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import classification_report

# 1. Veri yükle
df = pd.read_csv("goemotions_combined.csv")

# 2. Basit temizleme
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"[^a-zçğıöşü0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["text_clean"] = df["text"].apply(clean_text)

# 3. Duygu sütunlarını al ve önce sayısala çevir
emotion_cols = [c for c in df.columns if c not in ["text","author","link","text_clean"]]

# 3.a) Hangi sütunlar var, tipleri nedir kontrol edelim (isteğe bağlı)
print("Duygu sütunları ve tipleri:")
print(df[emotion_cols].dtypes)

# 3.b) String/NaN değerleri sayıya zorla, dönüştürülemeyenleri NaN olur, sonra 0 ile doldur
df[emotion_cols] = (
    df[emotion_cols]
      .apply(pd.to_numeric, errors="coerce")  # str→NaN
      .fillna(0)                              # NaN→0
)

# 3.c) Argmax ile label oluştur
df["label"] = df[emotion_cols].values.argmax(axis=1)

# 4. Split %80 train / %20 test
X_train, X_test, y_train, y_test = train_test_split(
    df["text_clean"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)

# 5. Pipeline
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ("clf",   LogisticRegression(max_iter=1000, multi_class="multinomial"))
])

# 6. Eğit & Değerlendir
print("Eğitim başladı...")
pipe.fit(X_train, y_train)
print("Eğitim tamamlandı.\nSonuçlar:")
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
