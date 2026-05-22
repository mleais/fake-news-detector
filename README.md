# 🔍 Fake News Detector — Yapay Zeka Destekli Sahte Haber Tespit Sistemi

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Doğal Dil İşleme (NLP) ve Makine Öğrenmesi teknikleri kullanarak haber metinlerinin gerçek mi yoksa sahte mi olduğunu tespit eden bir web uygulaması.

---

## 📸 Ekran Görüntüsü

> Uygulamayı çalıştırıp `screenshots/` klasörüne ekleyin.

---

## 🚀 Özellikler

- ✅ Metin girişi veya `.txt` dosyası yükleme ile analiz
- ✅ Otomatik metin ön işleme (tokenizasyon, stopword kaldırma, lemmatization)
- ✅ TF-IDF + Logistic Regression tabanlı sınıflandırma
- ✅ Gerçek zamanlı tahmin (FAKE / REAL) ve güven yüzdesi
- ✅ Kullanıcı dostu Streamlit arayüzü

---

## 🛠 Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| Arayüz | Streamlit |
| ML Modeli | scikit-learn (Logistic Regression) |
| Vektörizasyon | TF-IDF (1-gram + 2-gram) |
| Metin İşleme | NLTK |

---

## 📁 Proje Yapısı

```
fake_news_detector/
├── app.py                  # Streamlit ana uygulama
├── train.py                # Model eğitim scripti
├── requirements.txt        # Bağımlılıklar
├── models/
│   ├── model.py            # FakeNewsDetector sınıfı
│   └── fake_news_model.pkl # Eğitilmiş model (eğitimden sonra oluşur)
├── utils/
│   └── preprocessing.py    # TextPreprocessor sınıfı
└── data/                   # Veri setleri (Kaggle'dan indirilecek)
    ├── Fake.csv
    └── True.csv
```

---

## ⚙️ Kurulum

### 1. Repoyu klonlayın

```bash
git clone https://github.com/KULLANICI_ADINIZ/fake-news-detector.git
cd fake-news-detector
```

### 2. Sanal ortam oluşturun (önerilir)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
```

---

## 📊 Model Eğitimi

### Veri Seti İndirme

Kaggle'dan veri setini indirin:
👉 [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

`Fake.csv` ve `True.csv` dosyalarını `data/` klasörüne yerleştirin.

### Modeli Eğitin

```bash
python train.py --data_path ./data --test_size 0.2
```

Eğitim tamamlandığında model `models/fake_news_model.pkl` olarak kaydedilir.

**Beklenen Performans:**

| Metrik | Değer |
|---|---|
| Accuracy | ~0.98 |
| Precision | ~0.98 |
| Recall | ~0.98 |
| F1-Score | ~0.98 |

---

## 🌐 Uygulamayı Çalıştırma

```bash
streamlit run app.py
```

Tarayıcı otomatik açılır: `http://localhost:8501`

---

## 🎯 Kullanım

1. Analiz etmek istediğiniz haber metnini metin kutusuna yapıştırın veya `.txt` dosyası yükleyin
2. **"🔍 Analiz Et"** butonuna tıklayın
3. Sonucu ve güven yüzdesini görüntüleyin

---

## 📈 Model Mimarisi

```
Ham Metin
    ↓
TextPreprocessor (lowercase → URL temizle → tokenize → stopword kaldır → lemmatize)
    ↓
TF-IDF Vektörizer (max 50.000 özellik, 1-gram + 2-gram)
    ↓
Logistic Regression (C=1.0, class_weight='balanced')
    ↓
FAKE / REAL (+ olasılık skoru)
```

---


