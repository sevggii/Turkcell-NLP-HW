import pandas as pd
import re
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
import pickle

# Temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # sayıları kaldır
    text = text.translate(str.maketrans('', '', string.punctuation))  # noktalama kaldır
    text = text.strip()
    return text

input_dir = 'gyk-nlp/1234'
output_dir = 'gyk-nlp/processed_data'
max_len = 100
chunk_size = 10000

# Önce validation ve test dosyalarını oku, tokenizer'ı fit et
fit_files = ['validation.txt', 'test.txt']
all_articles = []
all_highlights = []
for fname in fit_files:
    path = os.path.join(input_dir, fname)
    df = pd.read_csv(path, sep=',', quotechar='"')
    df['article'] = df['article'].astype(str).apply(clean_text)
    df['highlights'] = df['highlights'].astype(str).apply(clean_text)
    all_articles.extend(df['article'].tolist())
    all_highlights.extend(df['highlights'].tolist())

article_tokenizer = Tokenizer(oov_token='<OOV>')
highlight_tokenizer = Tokenizer(oov_token='<OOV>')
article_tokenizer.fit_on_texts(all_articles)
highlight_tokenizer.fit_on_texts(all_highlights)

# Tokenizerları kaydet
with open(os.path.join(output_dir, 'article_tokenizer.pkl'), 'wb') as f:
    pickle.dump(article_tokenizer, f)
with open(os.path.join(output_dir, 'highlight_tokenizer.pkl'), 'wb') as f:
    pickle.dump(highlight_tokenizer, f)

# train.txt dosyasını chunk halinde işle
train_path = os.path.join(input_dir, 'train.txt')
article_chunks = []
highlight_chunks = []
reader = pd.read_csv(train_path, sep=',', quotechar='"', chunksize=chunk_size)
for chunk in reader:
    chunk['article'] = chunk['article'].astype(str).apply(clean_text)
    chunk['highlights'] = chunk['highlights'].astype(str).apply(clean_text)
    article_seq = article_tokenizer.texts_to_sequences(chunk['article'])
    highlight_seq = highlight_tokenizer.texts_to_sequences(chunk['highlights'])
    article_pad = pad_sequences(article_seq, maxlen=max_len, padding='post', truncating='post')
    highlight_pad = pad_sequences(highlight_seq, maxlen=max_len, padding='post', truncating='post')
    article_chunks.append(article_pad)
    highlight_chunks.append(highlight_pad)

# Tüm chunk'ları birleştir
article_all = np.vstack(article_chunks)
highlight_all = np.vstack(highlight_chunks)

# Kaydet
np.savez(os.path.join(output_dir, 'train_processed.npz'), article=article_all, highlights=highlight_all)

print('train.txt için ön işleme ve padding tamamlandı!') 