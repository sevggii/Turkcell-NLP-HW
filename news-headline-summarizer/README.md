##Merhabalar, bu ödev Sevgi Targay ve Büşranur Çevik tarafından hazırlanmıştır

# Haber Başlıklarından Otomatik Özetleme (NLP Ödevi)

## Amaç
CNN/DailyMail veri seti üzerinde, transformer tabanlı (T5-small) bir model ile haber metinlerinden otomatik özet çıkaran bir sistem geliştirmek.

## Adımlar
1. **Veri Ön İşleme:**
   - Metin temizliği (noktalama, küçük harfe çevirme, sayı temizliği)
   - Maksimum uzunluklarla truncation ve padding
2. **Model Kurulumu:**
   - Huggingface Transformers ile T5-small modeli
3. **Eğitim:**
   - Trainer ile 3 epoch hızlı eğitim
4. **Değerlendirme:**
   - ROUGE-L metriği ile özet kalitesi ölçümü
   - En az 5 örnek giriş ve özet karşılaştırması

## Kullanılan Model ve Hiperparametreler
- Model: T5-small
- max_input: 256
- max_target: 64
- Epoch: 3
- Batch size: 4
- Learning rate: 5e-5

## Geliştirme Süreci
- Veri ön işleme, model kurulumu, eğitim ve değerlendirme adımları notebook'ta sırasıyla uygulanmıştır.
- Küçük bir veri örneğiyle hızlı prototip alınmıştır.
- ROUGE-L metriği ile özet kalitesi ölçülmüştür.

## Çalıştırma
1. `summarization_pipeline.ipynb` dosyasını açın.
2. Tüm hücreleri sırasıyla çalıştırın.
3. Sonuçlar ve örnek çıktılar notebook sonunda yer almaktadır.

## Not
- Büyük veriyle çalışmak donanımda sorun çıkarabilir. Örneklem alınmıştır.
- Tüm kodlar ve açıklamalar Türkçedir. 