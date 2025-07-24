import pandas as pd
import numpy as np
import re
import string
import os
from transformers import T5Tokenizer

def clean_text(text):
    return re.sub(r'\d+', '', text.lower().translate(str.maketrans('', '', string.punctuation))).strip()

def process_data_for_t5():
    print("T5 modeli için veri işleme başlıyor...")
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    cnn_dir = "cnn_dailymail"
    output_dir = cnn_dir
    os.makedirs(output_dir, exist_ok=True)
    for split, max_input, max_target in [
        ("train.csv", 512, 128),
        ("validation.csv", 512, 128),
        ("test.csv", 512, 128)
    ]:
        path = os.path.join(cnn_dir, split)
        if not os.path.exists(path):
            continue
        print(f"{split} işleniyor...")
        df = pd.read_csv(path)
        df['article'] = df['article'].astype(str).apply(clean_text)
        df['highlights'] = df['highlights'].astype(str).apply(clean_text)
        df['input_text'] = 'summarize: ' + df['article']
        encodings = tokenizer(
            df['input_text'].tolist(),
            truncation=True,
            padding=True,
            max_length=max_input,
            return_tensors='np'
        )
        target_encodings = tokenizer(
            df['highlights'].tolist(),
            truncation=True,
            padding=True,
            max_length=max_target,
            return_tensors='np'
        )
        np.savez(
            os.path.join(output_dir, f'{split.replace(".csv", "_processed_t5.npz")}'),
            input_ids=encodings['input_ids'],
            attention_mask=encodings['attention_mask'],
            labels=target_encodings['input_ids']
        )
        print(f"{split} kaydedildi: {len(df)} örnek")

if __name__ == "__main__":
    process_data_for_t5() 