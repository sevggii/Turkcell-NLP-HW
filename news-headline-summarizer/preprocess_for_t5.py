import pandas as pd
import re
import string
import os
import numpy as np
from transformers import T5Tokenizer
import json

def clean_text(text):
    """Metni temizle"""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # sayıları kaldır
    text = text.translate(str.maketrans('', '', string.punctuation))  # noktalama kaldır
    text = text.strip()
    return text

def process_data_for_t5():
    """T5 modeli için veriyi işle"""
    print("T5 modeli için veri işleme başlıyor...")
    
    # T5 tokenizer'ı yükle
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    # Data yolları
    cnn_dir = "cnn_dailymail"
    train_path = os.path.join(cnn_dir, "train.csv")
    val_path = os.path.join(cnn_dir, "validation.csv")
    test_path = os.path.join(cnn_dir, "test.csv")
    
    # Output dizini
    output_dir = "cnn_dailymail"
    os.makedirs(output_dir, exist_ok=True)
    
    # Train data işle
    if os.path.exists(train_path):
        print("Train data işleniyor...")
        train_df = pd.read_csv(train_path)
        
        # Metinleri temizle
        train_df['article'] = train_df['article'].astype(str).apply(clean_text)
        train_df['highlights'] = train_df['highlights'].astype(str).apply(clean_text)
        
        # T5 formatına çevir (summarize: prefix ekle)
        train_df['input_text'] = 'summarize: ' + train_df['article']
        
        # Tokenize ve kaydet
        train_encodings = tokenizer(
            train_df['input_text'].tolist(),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        target_encodings = tokenizer(
            train_df['highlights'].tolist(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Numpy array'e çevir
        train_input_ids = train_encodings['input_ids'].numpy()
        train_attention_mask = train_encodings['attention_mask'].numpy()
        train_labels = target_encodings['input_ids'].numpy()
        
        # Kaydet
        np.savez(
            os.path.join(output_dir, 'train_processed_t5.npz'),
            input_ids=train_input_ids,
            attention_mask=train_attention_mask,
            labels=train_labels
        )
        
        print(f"Train data kaydedildi: {len(train_df)} samples")
    
    # Validation data işle
    if os.path.exists(val_path):
        print("Validation data işleniyor...")
        val_df = pd.read_csv(val_path)
        
        # Metinleri temizle
        val_df['article'] = val_df['article'].astype(str).apply(clean_text)
        val_df['highlights'] = val_df['highlights'].astype(str).apply(clean_text)
        
        # T5 formatına çevir
        val_df['input_text'] = 'summarize: ' + val_df['article']
        
        # Tokenize ve kaydet
        val_encodings = tokenizer(
            val_df['input_text'].tolist(),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        val_target_encodings = tokenizer(
            val_df['highlights'].tolist(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Numpy array'e çevir
        val_input_ids = val_encodings['input_ids'].numpy()
        val_attention_mask = val_encodings['attention_mask'].numpy()
        val_labels = val_target_encodings['input_ids'].numpy()
        
        # Kaydet
        np.savez(
            os.path.join(output_dir, 'validation_processed_t5.npz'),
            input_ids=val_input_ids,
            attention_mask=val_attention_mask,
            labels=val_labels
        )
        
        print(f"Validation data kaydedildi: {len(val_df)} samples")
    
    # Test data işle
    if os.path.exists(test_path):
        print("Test data işleniyor...")
        test_df = pd.read_csv(test_path)
        
        # Metinleri temizle
        test_df['article'] = test_df['article'].astype(str).apply(clean_text)
        test_df['highlights'] = test_df['highlights'].astype(str).apply(clean_text)
        
        # T5 formatına çevir
        test_df['input_text'] = 'summarize: ' + test_df['article']
        
        # Tokenize ve kaydet
        test_encodings = tokenizer(
            test_df['input_text'].tolist(),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        test_target_encodings = tokenizer(
            test_df['highlights'].tolist(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Numpy array'e çevir
        test_input_ids = test_encodings['input_ids'].numpy()
        test_attention_mask = test_encodings['attention_mask'].numpy()
        test_labels = test_target_encodings['input_ids'].numpy()
        
        # Kaydet
        np.savez(
            os.path.join(output_dir, 'test_processed_t5.npz'),
            input_ids=test_input_ids,
            attention_mask=test_attention_mask,
            labels=test_labels
        )
        
        print(f"Test data kaydedildi: {len(test_df)} samples")
    
    # Tokenizer'ı kaydet
    tokenizer.save_pretrained(os.path.join(output_dir, 't5_tokenizer'))
    
    print("T5 için veri işleme tamamlandı!")
    print("Oluşturulan dosyalar:")
    print("- train_processed_t5.npz")
    print("- validation_processed_t5.npz") 
    print("- test_processed_t5.npz")
    print("- t5_tokenizer/")

if __name__ == "__main__":
    process_data_for_t5() 