#!/usr/bin/env python3
"""
T5 Model Kurulum Test Script'i
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

def test_model_loading():
    """T5 modelini yüklemeyi test et"""
    print("=== T5 Model Kurulum Testi ===")
    
    try:
        # Model ve tokenizer yükle
        print("T5-small modeli yükleniyor...")
        model_name = "t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        print(f"✅ Model başarıyla yüklendi: {model_name}")
        print(f"📊 Model parametre sayısı: {model.num_parameters():,}")
        print(f"🔤 Tokenizer vocab boyutu: {tokenizer.vocab_size}")
        
        # GPU kontrolü
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Kullanılan cihaz: {device}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Basit test
        test_text = "summarize: This is a test article for the T5 model."
        inputs = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True)
        
        print(f"✅ Tokenization testi başarılı")
        print(f"📝 Test input shape: {inputs['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        return False

def test_data_processing():
    """Veri işleme testi"""
    print("\n=== Veri İşleme Testi ===")
    
    cnn_dir = "cnn_dailymail"
    train_path = os.path.join(cnn_dir, "train_processed_t5.npz")
    val_path = os.path.join(cnn_dir, "validation_processed_t5.npz")
    
    if os.path.exists(train_path):
        print(f"✅ Train data mevcut: {train_path}")
    else:
        print(f"❌ Train data bulunamadı: {train_path}")
        print("💡 Çözüm: python preprocess_for_t5.py çalıştırın")
    
    if os.path.exists(val_path):
        print(f"✅ Validation data mevcut: {val_path}")
    else:
        print(f"❌ Validation data bulunamadı: {val_path}")
        print("💡 Çözüm: python preprocess_for_t5.py çalıştırın")

def test_requirements():
    """Gerekli kütüphaneleri test et"""
    print("\n=== Kütüphane Testi ===")
    
    required_packages = [
        'torch',
        'transformers', 
        'numpy',
        'pandas',
        'tqdm'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} yüklü")
        except ImportError:
            print(f"❌ {package} yüklü değil")
            print(f"💡 Çözüm: pip install {package}")

def main():
    """Ana test fonksiyonu"""
    print("🚀 T5 News Summarization Model Kurulum Testi")
    print("=" * 50)
    
    # Kütüphane testi
    test_requirements()
    
    # Model testi
    model_ok = test_model_loading()
    
    # Data testi
    test_data_processing()
    
    print("\n" + "=" * 50)
    if model_ok:
        print("🎉 Model kurulumu başarılı!")
        print("📚 Training başlatmak için: python train_model.py")
    else:
        print("⚠️  Model kurulumunda sorun var!")
        print("🔧 Lütfen hataları kontrol edin ve gerekli kütüphaneleri yükleyin")

if __name__ == "__main__":
    main() 