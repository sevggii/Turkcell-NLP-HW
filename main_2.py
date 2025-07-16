import re
import pandas as pd
from sklearn.model_selection      import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model        import LogisticRegression
from sklearn.pipeline            import Pipeline
from sklearn.metrics             import classification_report

# 1. Veri yükle
df = pd.read_csv("goemotions_combined.csv")

# 2. Basit temizleme
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"[^a-zçğıöşü0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["text_clean"] = df["text"].apply(clean_text)

# 3. Metadata sütunları (hariç tut)
meta = [
    "id", "subreddit", "link_id", "parent_id",
    "created_utc", "rater_id", "example_very_unclear",
    "author", "text", "text_clean"
]

# 4. Sadece gerçek duygu sütunlarını seç
emotion_cols = [c for c in df.columns if c not in meta]
print("Kullanılacak duygu sütunları:", emotion_cols)

# 5. Dönüşüm: str/NaN → sayısal → NaN’ları 0 yap
df[emotion_cols] = (
    df[emotion_cols]
      .apply(pd.to_numeric, errors="coerce")
      .fillna(0)
)

# 6. Çok-sınıflı label (argmax)
df["label"] = df[emotion_cols].values.argmax(axis=1)

# 7. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text_clean"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)

# 8. Pipeline tanımı
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ("clf",   LogisticRegression(max_iter=1000, multi_class="multinomial"))
])

# 9. Eğit & değerlendir
print("\n=== MODEL EĞİTİMİ VE DEĞERLENDİRME ===")
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# 10. İlk 10 örnekten duygu tahmini yap
print("\n=== İLK 10 ÖRNEK METİN VE TAHMİNİ ===")
sample_df = df.iloc[:10]
for idx, row in sample_df.iterrows():
    text = row["text"]
    clean = row["text_clean"]
    label_idx = pipe.predict([clean])[0]
    emotion = emotion_cols[label_idx]
    print(f"{idx+1}. \"{text}\"  →  {emotion}")
