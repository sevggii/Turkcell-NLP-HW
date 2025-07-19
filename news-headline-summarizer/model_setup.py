import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import Trainer, TrainingArguments
import numpy as np
import pickle
import os
from torch.utils.data import Dataset, DataLoader

class NewsSummarizationDataset(Dataset):
    def __init__(self, articles, highlights, tokenizer, max_length=512):
        self.articles = articles
        self.highlights = highlights
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        article = str(self.articles[idx])
        highlight = str(self.highlights[idx])
        
        # Tokenize input and target
        inputs = self.tokenizer(
            article,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            highlight,
            max_length=128,  # Shorter for summaries
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten()
        }

def load_processed_data(data_path):
    """Processed .npz dosyalarını yükle"""
    data = np.load(data_path)
    return data['article'], data['highlights']

def setup_model():
    """T5-small modelini kur ve yapılandır"""
    print("T5-small modeli yükleniyor...")
    
    # Model ve tokenizer'ı yükle
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Özel tokenlar ekle (gerekirse)
    special_tokens = ['<START>', '<END>']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"Model yüklendi: {model_name}")
    print(f"Model parametre sayısı: {model.num_parameters():,}")
    
    return model, tokenizer

def create_datasets(tokenizer, train_data_path, val_data_path):
    """Train ve validation dataset'lerini oluştur"""
    print("Dataset'ler oluşturuluyor...")
    
    # Train data yükle
    train_articles, train_highlights = load_processed_data(train_data_path)
    
    # Validation data yükle
    val_articles, val_highlights = load_processed_data(val_data_path)
    
    # Dataset'leri oluştur
    train_dataset = NewsSummarizationDataset(
        train_articles, 
        train_highlights, 
        tokenizer
    )
    
    val_dataset = NewsSummarizationDataset(
        val_articles, 
        val_highlights, 
        tokenizer
    )
    
    print(f"Train dataset boyutu: {len(train_dataset)}")
    print(f"Validation dataset boyutu: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def setup_training_args(output_dir="./model_output"):
    """Training argümanlarını yapılandır"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Düşük donanım için
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,  # Memory kullanımını azalt
        gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
        fp16=True,  # Mixed precision training
        report_to=None,  # Wandb gibi logging'i kapat
    )
    
    return training_args

if __name__ == "__main__":
    # Model kurulumu
    model, tokenizer = setup_model()
    
    # Dataset'leri oluştur (eğer processed data varsa)
    cnn_dir = "cnn_dailymail"
    train_path = os.path.join(cnn_dir, "train_processed.npz")
    val_path = os.path.join(cnn_dir, "validation_processed.npz")
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        train_dataset, val_dataset = create_datasets(tokenizer, train_path, val_path)
        
        # Training argümanlarını ayarla
        training_args = setup_training_args()
        
        # Trainer oluştur
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
        )
        
        print("Model kurulumu tamamlandı!")
        print("Training başlatmak için: trainer.train()")
        
    else:
        print("Processed data dosyaları bulunamadı!")
        print("Önce preprocess_and_pad.py çalıştırın.") 