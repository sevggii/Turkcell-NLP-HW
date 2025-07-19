import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments
import numpy as np
import os
from t5_dataset import T5SummarizationDataset, create_t5_datasets

def train_model():
    """Modeli eğit"""
    print("=== News Headline Summarization Model Training ===")
    
    # Model ve tokenizer yükle
    print("T5-small modeli yükleniyor...")
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    print(f"Model yüklendi: {model_name}")
    print(f"Model parametre sayısı: {model.num_parameters():,}")
    
    # Data yolları
    cnn_dir = "cnn_dailymail"
    train_path = os.path.join(cnn_dir, "train_processed_t5.npz")
    val_path = os.path.join(cnn_dir, "validation_processed_t5.npz")
    
    # Data kontrolü
    if not os.path.exists(train_path):
        print(f"Train data bulunamadı: {train_path}")
        print("Önce preprocess_for_t5.py çalıştırın!")
        return
    
    if not os.path.exists(val_path):
        print(f"Validation data bulunamadı: {val_path}")
        print("Önce preprocess_for_t5.py çalıştırın!")
        return
    
    print("Data dosyaları yükleniyor...")
    
    # Dataset'leri oluştur
    train_dataset, val_dataset, _ = create_t5_datasets(train_path, val_path)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Training argümanları
    training_args = TrainingArguments(
        output_dir="./t5_summarization_model",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Düşük donanım için
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=8,  # Effective batch size = 2 * 8 = 16
        fp16=True,  # Mixed precision
        report_to=None,
        remove_unused_columns=False,
    )
    
    # Trainer oluştur
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    print("Training başlıyor...")
    print("Bu işlem uzun sürebilir. Lütfen bekleyin...")
    
    # Training başlat
    trainer.train()
    
    # Modeli kaydet
    model_save_path = "./t5_summarization_model/final_model"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print(f"Model kaydedildi: {model_save_path}")
    print("Training tamamlandı!")

def test_model(model_path="./t5_summarization_model/final_model"):
    """Eğitilmiş modeli test et"""
    if not os.path.exists(model_path):
        print(f"Model bulunamadı: {model_path}")
        return
    
    print("Model test ediliyor...")
    
    # Model ve tokenizer yükle
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # Test metni
    test_article = """
    The new artificial intelligence system developed by researchers at Stanford University 
    has shown remarkable results in natural language processing tasks. The system, which 
    uses advanced transformer architecture, achieved state-of-the-art performance on 
    multiple benchmark datasets. According to the research team, this breakthrough 
    could revolutionize how we interact with computers and process large amounts of text data.
    """
    
    # Tokenize
    inputs = tokenizer(
        test_article,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=128,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    
    # Decode
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("Test Article:")
    print(test_article.strip())
    print("\nGenerated Summary:")
    print(summary)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_model()
    else:
        train_model() 