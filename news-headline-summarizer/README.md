# News Headline Summarization with T5

Bu proje, CNN/DailyMail veri seti kullanarak haber başlığı özetleme modeli eğitmek için tasarlanmıştır. T5-small modeli kullanılarak düşük donanım gereksinimleriyle çalışacak şekilde optimize edilmiştir.

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Veri Ön İşleme (T5 için)
T5 modeli için veri ön işleme:
```bash
python preprocess_for_t5.py
```

### 2. Model Kurulum Testi
Kurulumu test etmek için:
```bash
python test_setup.py
```

### 3. Model Eğitimi
Modeli eğitmek için:
```bash
python train_model.py
```

### 4. Model Testi
Eğitilmiş modeli test etmek için:
```bash
python train_model.py test
```

## Model Özellikleri

- **Model**: T5-small (60M parametre)
- **Input Length**: 512 token
- **Output Length**: 128 token
- **Batch Size**: 2 (effective 16 with gradient accumulation)
- **Mixed Precision**: FP16
- **Optimization**: Düşük donanım için optimize edilmiş

## Dosya Yapısı

```
news-headline-summarizer/
├── cnn_dailymail/
│   ├── preprocess_and_pad.py       # Eski veri ön işleme
│   ├── train_processed.npz         # Eski işlenmiş veri
│   ├── validation_processed.npz    # Eski işlenmiş veri
│   ├── test_processed.npz          # Eski işlenmiş veri
│   ├── train_processed_t5.npz      # T5 için işlenmiş train verisi
│   ├── validation_processed_t5.npz # T5 için işlenmiş validation verisi
│   ├── test_processed_t5.npz       # T5 için işlenmiş test verisi
│   └── t5_tokenizer/               # T5 tokenizer
├── preprocess_for_t5.py            # T5 için veri ön işleme
├── t5_dataset.py                   # T5 dataset sınıfı
├── train_model.py                  # Training script'i
├── requirements.txt                # Gerekli kütüphaneler
└── README.md                      # Bu dosya
```

## Donanım Gereksinimleri

- **Minimum RAM**: 8GB
- **GPU**: 4GB+ VRAM (önerilen)
- **CPU**: 4+ çekirdek
- **Disk**: 5GB+ boş alan

## Performans Optimizasyonları

1. **Gradient Accumulation**: Effective batch size artırıldı
2. **Mixed Precision**: FP16 kullanımı
3. **Memory Optimization**: Pin memory kapatıldı
4. **Small Model**: T5-small kullanımı

## Sorun Giderme

### CUDA Out of Memory
- `per_device_train_batch_size` değerini düşürün
- `gradient_accumulation_steps` değerini artırın

### Slow Training
- `fp16=True` olduğundan emin olun
- GPU kullanımını kontrol edin

### Data Loading Issues
- `preprocess_and_pad.py` dosyasının çalıştırıldığından emin olun
- `.npz` dosyalarının mevcut olduğunu kontrol edin 