import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import numpy as np
from t5_dataset import create_t5_datasets

def main():
    print("=== Eğitim: News Headline Summarization ===")
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Veri yolları
    cnn_dir = "cnn_dailymail"
    train_path = os.path.join(cnn_dir, "train_processed_t5.npz")
    val_path = os.path.join(cnn_dir, "validation_processed_t5.npz")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("İşlenmiş veri dosyaları bulunamadı! Önce preprocess_for_t5.py çalıştırılmalı.")
        return

    # Datasetleri oluştur
    train_dataset, val_dataset, _ = create_t5_datasets(train_path, val_path)

    # Eğitim argümanları
    training_args = TrainingArguments(
        output_dir="./t5_summarization_model",
        num_train_epochs=3,  # Hızlı prototip için 3 epoch
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
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
        gradient_accumulation_steps=8,
        fp16=True,
        report_to=None,
        remove_unused_columns=False,
    )

    # Trainer ile eğitim
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    print("Eğitim başlıyor...")
    trainer.train()

    # Modeli kaydet
    model_save_path = "./t5_summarization_model/final_model"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model kaydedildi: {model_save_path}")
    print("Eğitim tamamlandı!")

if __name__ == "__main__":
    main() 