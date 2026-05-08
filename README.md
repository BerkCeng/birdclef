# BirdCLEF Kuş Sesi Sınıflandırma

Kuş sesi kayıtlarından üretilen Mel-spektrogram görüntüleri üzerinde CNN (ResNet34) ve CRNN (ResNet18 + BiLSTM) mimarilerini kullanarak tür sınıflandırması yapan PyTorch tabanlı derin öğrenme projesi.

## Proje Özeti

Ham ses dosyaları (`32 kHz` örnekleme hızı, `5 saniye` pencere uzunluğu, `128` Mel bandı, `2048` FFT boyutu ve `512` hop length) log-Mel spektrogram görüntülerine (`224x224` RGB) dönüştürülür. Elde edilen spektrogramlar üzerinde uzamsal öznitelikler CNN omurgası ile çıkarılır, CRNN mimarisinde ise frekans-zaman eksenindeki ardışık bağımlılıklar çift yönlü LSTM katmanı ile modellenir. Model kararlarının frekans-zaman düzlemindeki aktivasyon bölgeleri Grad-CAM ile görselleştirilir.

## Deneysel Sonuçlar

264 kuş türü ve toplam 137.767 spektrogram örneği üzerinde eğitilen CRNN (`ResNet18 + BiLSTM`) modelinin değerlendirme sonuçları:

| Model | Backbone | Sınıf Sayısı | Örnek Sayısı | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BirdCRNN | ResNet18 + BiLSTM | 264 | 137.767 | 0.99 | 0.99 | 0.98 | 0.98 | 0.99 |

## Dosya Yapısı

```text
birdclef/
├── data/
│   ├── raw/                  # Eğitim/değerlendirme spektrogram veri seti (sınıf bazlı klasörler)
│   ├── bird_voice_image/     # Ham sesten dönüştürülen spektrogram çıktıları
│   └── test_images/          # Tekil/toplu tahmin için test görselleri
├── bird_voice/               # Ham ses dosyaları (.wav, .mp3, .ogg, .flac)
├── models/                   # Eğitilmiş model ağırlıkları (.pth)
├── outputs/
│   ├── logs/                 # Eğitim grafikleri, karmaşıklık matrisi ve sınıflandırma raporu
│   ├── batch_predictions/    # Toplu tahmin görselleri
│   └── gradcam/              # Grad-CAM aktivasyon çıktıları
├── src/
│   ├── models/
│   │   ├── cnn_model.py      # timm tabanlı CNN modeli
│   │   └── crnn_model.py     # ResNet18 + BiLSTM tabanlı CRNN modeli
│   ├── config.py             # Dizin yolları, model ve sinyal işleme hiperparametreleri
│   ├── data_loader.py        # PyTorch Dataset ve DataLoader tanımları
│   ├── audio_utils.py        # Ses segmentasyonu ve Mel-spektrogram dönüşümü
│   ├── prepare_data.py       # Ham ses arşivinden toplu spektrogram üretimi
│   ├── train.py              # Model eğitim döngüsü
│   ├── evaluate.py           # Karmaşıklık matrisi ve sınıflandırma metrikleri hesabı
│   ├── predict.py            # Test klasöründeki spektrogramlar için toplu tahmin
│   ├── audio_inference.py    # Ham ses dosyası üzerinden segment bazlı çıkarım
│   ├── gradcam.py            # Grad-CAM ısı haritası hesaplama ve görselleştirme
│   ├── utils.py              # Grafik çizimi ve raporlama yardımcı fonksiyonları
│   └── app.py                # Streamlit tabanlı spektrogram analiz arayüzü
└── requirements.txt
```

## Kurulum

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Kullanım

### 1. Ham Seslerden Spektrogram Üretimi

`bird_voice/<tur_adi>/` altında yer alan ses kayıtlarını 5 saniyelik segmentlere bölüp `data/bird_voice_image/` dizinine kaydetmek için:

```bash
python3 src/prepare_data.py
```

### 2. Model Eğitimi

Hiperparametreler ve model seçimi (`MODEL_TYPE = "crnn"` veya `"cnn"`) `src/config.py` dosyasından yapılandırılır:

```bash
python3 src/train.py
```

### 3. Değerlendirme

Kayıtlı model ağırlıklarını kullanarak karmaşıklık matrisi ve sınıflandırma raporunu `outputs/logs/` dizinine üretmek için:

```bash
python3 src/evaluate.py
```

### 4. Tahmin ve Çıkarım

`data/test_images/` dizinindeki spektrogram görüntüleri üzerinde toplu tahmin çalıştırmak için:

```bash
python3 src/predict.py
```

Tek bir ham ses dosyası üzerinde 5 saniyelik pencereler halinde çıkarım yapmak için:

```bash
python3 src/audio_inference.py <ses_dosyasi_yolu>
```

### 5. Web Arayüzü

Spektrogram yükleme, sınıf olasılıklarını görüntüleme ve Grad-CAM aktivasyon haritası oluşturma arayüzünü başlatmak için:

```bash
streamlit run src/app.py
```
