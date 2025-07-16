
# Kullanıcıdan gelen mesaja anlamlı ve farklı cevaplar döndürülmektedir.
# Kendi modelinizi veya daha gelişmiş cevap mantıklarını kolayca entegre edebilirsiniz.
#
# Transfer öğrenme (Transfer Learning): Daha önce büyük veriyle eğitilmiş bir modeli alıp, kendi küçük veri setimize uyarlayarak kullanmak anlamına gelir. Böylece sıfırdan model eğitmek yerine, hazır bir beyni kendi işimize adapte etmiş oluruz.
# Derin öğrenme (Deep Learning): Hiçbir ön bilgi olmadan, modeli en baştan kendi verimizle eğitmek ve sıfırdan bir yapay zeka oluşturmak demektir.

# Gerekli kütüphaneleri içe aktar
import gradio as gr

# Chatbot fonksiyonu: Kullanıcıdan gelen girdiye yanıt üretir
# Burada örnek olarak anahtar kelimeye göre farklı cevaplar döndürülmektedir
# Daha gelişmiş bir model veya API ile kolayca değiştirilebilir

def chatbot_cevapla(kullanici_mesaji):
    """
    Kullanıcıdan gelen mesaja göre chatbot'un vereceği yanıtı üretir.
    """
    # Mesaj boşsa uyarı ver
    if not kullanici_mesaji.strip():
        return "Lütfen bir mesaj girin."
    # Anahtar kelimeye göre örnek cevaplar
    if "merhaba" in kullanici_mesaji.lower():
        return "Merhaba! Sana nasıl yardımcı olabilirim?"
    elif "nasılsın" in kullanici_mesaji.lower():
        return "Ben bir yapay zekayım, duygularım yok ama yardımcı olmak için buradayım!"
    elif "hava" in kullanici_mesaji.lower():
        return "Bugün hava oldukça güzel görünüyor!"
    elif "teşekkür" in kullanici_mesaji.lower():
        return "Rica ederim, her zaman yardımcı olmaktan mutluluk duyarım!"
    else:
        # Bilinmeyen sorulara genel cevap
        return f"Sorduğun: {kullanici_mesaji}\nCevabım: Şu anda bu konuda bir bilgim yok, ama öğrenebilirim!"

# Gradio arayüzünü oluştur
# Kullanıcıdan metin alır ve chatbot_cevapla fonksiyonunun çıktısını gösterir
chat_arayuz = gr.Interface(
    fn=chatbot_cevapla,  # Çalıştırılacak fonksiyon
    inputs=gr.Textbox(lines=2, placeholder="Bir mesaj yazın..."),  # Kullanıcıdan metin girişi
    outputs="text",  # Metin çıktısı
    title="Özgün Chatbot Arayüzü",  # Arayüz başlığı
    description="Mesajınızı yazın, chatbot anlamlı bir cevap versin. (Bu arayüz ödev için özgün hazırlanmıştır.)"
)

# Uygulamayı başlat
if __name__ == "__main__":
    chat_arayuz.launch(share=True) 