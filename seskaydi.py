import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

def record_audio(duration, fs=48000, filename="output.wav"):
    """
    Mikrofondan belirtilen süre ve frekansta ses kaydı alır.
    
    Parametreler:
        duration (int veya float): Kayıt süresi (saniye cinsinden)
        fs (int): Örnekleme hızı (Hz - saniyedeki örnek sayısı). İstenildiği üzere varsayılan 48000.
        filename (str): Kaydedilecek wav ve dosya adı.
    """
    print(f"🎤 {duration} saniye boyunca kayıt alınıyor... Lütfen konuşun.")
    print(f"⚙️ Örnekleme Hızı (Sample Rate): {fs} Hz")
    
    # sd.rec ile kaydı başlat (channels=1 ile mono kayıt alıyoruz)
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    
    # Kaydın tamamen bitmesini bekle
    sd.wait()
    
    print("✅ Kayıt tamamlandı. Dosyaya kaydediliyor...")
    
    # numpy array'ini .wav dosyası olarak kaydet
    write(filename, fs, recording)
    
    print(f"💾 Kayıt başarıyla '{filename}' olarak kaydedildi.")

if __name__ == "__main__":
    # Örnek kullanım: 5 saniyelik bir ses kaydı alma
    kayit_suresi = 5  # İsteğe göre kayıt süresini değiştirebilirsiniz
    dosya_ismi = "ornek_kayit.wav"
    
    try:
        record_audio(duration=kayit_suresi, fs=48000, filename=dosya_ismi)
    except Exception as e:
        print(f"❌ Bir hata oluştu: {e}")
        print("Lütfen mikrofonunuzun bağlı olduğundan ve mikrofon erişim izninizin olduğundan emin olun.")
