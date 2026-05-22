"""
Model Eğitim Scripti - Fake News Detection
============================================
Kaggle veri setini kullanarak modeli eğitir ve kaydeder.

Kullanım:
    python train.py --data_path ./data/news.csv --test_size 0.2
"""

import os
import sys
import argparse
import logging
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Proje modüllerini import et
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.model import FakeNewsDetector
from utils.preprocessing import TextPreprocessor

# Logging konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Veri seti yükleme ve hazırlama sınıfı.

    Desteklenen formatlar:
    - Fake and Real News Dataset (Kaggle)
    - LIAR Dataset
    - Fake News Detection Dataset
    """

    @staticmethod
    def load_fake_real_dataset(data_dir: str) -> pd.DataFrame:
        """
        Kaggle Fake and Real News Dataset'ini yükler.
        Beklenen dosyalar: Fake.csv, True.csv

        Args:
            data_dir: Veri dizini yolu

        Returns:
            text ve label sütunları olan DataFrame
        """
        fake_path = os.path.join(data_dir, 'Fake.csv')
        true_path = os.path.join(data_dir, 'True.csv')

        if not os.path.exists(fake_path) or not os.path.exists(true_path):
            raise FileNotFoundError(
                f"Veri dosyaları bulunamadı: {fake_path} ve {true_path}\n"
                "Lütfen Kaggle'dan 'Fake and Real News Dataset' indirip data/ klasörüne ekleyin."
            )

        logger.info("Fake.csv yükleniyor...")
        fake_df = pd.read_csv(fake_path)
        fake_df['label'] = 1  # 1 = Sahte
        fake_df['label_text'] = 'FAKE'

        logger.info("True.csv yükleniyor...")
        true_df = pd.read_csv(true_path)
        true_df['label'] = 0  # 0 = Gerçek
        true_df['label_text'] = 'REAL'

        # Birleştir
        df = pd.concat([fake_df, true_df], ignore_index=True)

        # Başlık + içerik birleştir (daha iyi özellik)
        if 'title' in df.columns and 'text' in df.columns:
            df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        elif 'text' in df.columns:
            df['full_text'] = df['text'].fillna('')
        else:
            raise ValueError("Veri setinde 'text' sütunu bulunamadı.")

        logger.info(f"Toplam: {len(df)} örnek | Fake: {fake_df.shape[0]} | Real: {true_df.shape[0]}")
        return df[['full_text', 'label', 'label_text']]

    @staticmethod
    def load_single_csv(csv_path: str, text_col: str = 'text', label_col: str = 'label') -> pd.DataFrame:
        """
        Tek CSV dosyasını yükler.

        Args:
            csv_path: CSV dosya yolu
            text_col: Metin sütun adı
            label_col: Etiket sütun adı

        Returns:
            Hazırlanmış DataFrame
        """
        df = pd.read_csv(csv_path)

        if text_col not in df.columns:
            raise ValueError(f"Sütun bulunamadı: '{text_col}'. Mevcut sütunlar: {list(df.columns)}")

        df = df.rename(columns={text_col: 'full_text', label_col: 'label'})
        df = df.dropna(subset=['full_text'])
        return df[['full_text', 'label']]


def train_model(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    """
    Ana model eğitim fonksiyonu.

    Args:
        data_path: Veri dizini veya CSV dosyası yolu
        test_size: Test seti oranı
        random_state: Rastgele tohum
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("FAKE NEWS DETECTOR - MODEL EĞİTİMİ BAŞLIYOR")
    logger.info("=" * 60)

    # 1. Veri yükle
    logger.info("Adım 1/5: Veri yükleniyor...")
    try:
        if os.path.isdir(data_path):
            df = DataLoader.load_fake_real_dataset(data_path)
        else:
            df = DataLoader.load_single_csv(data_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # 2. Ön işleme
    logger.info("Adım 2/5: Metin ön işleme uygulanıyor...")
    preprocessor = TextPreprocessor(
        use_lemmatization=True,
        remove_stopwords=True
    )
    df['processed_text'] = preprocessor.preprocess_batch(df['full_text'].tolist())

    # Boş metinleri filtrele
    df = df[df['processed_text'].str.strip() != '']
    logger.info(f"Ön işleme sonrası: {len(df)} örnek")

    # 3. Eğitim/Test ayrımı
    logger.info("Adım 3/5: Veri seti bölünüyor...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'].tolist(),
        df['label'].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=df['label'].tolist()
    )
    logger.info(f"Eğitim: {len(X_train)} | Test: {len(X_test)}")

    # 4. Model eğit
    logger.info("Adım 4/5: Model eğitiliyor...")
    detector = FakeNewsDetector(
        max_features=50000,
        ngram_range=(1, 2),
        max_iter=1000,
        C=1.0
    )
    detector.train(X_train, y_train)

    # 5. Değerlendirme
    logger.info("Adım 5/5: Model değerlendiriliyor...")
    metrics = detector.evaluate(X_test, y_test)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANS METRİKLERİ")
    print("=" * 60)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    print("\nDetaylı Rapor:")
    print(metrics['report'])

    # Model kaydet
    detector.save()

    elapsed = time.time() - start_time
    logger.info(f"Eğitim tamamlandı! Süre: {elapsed:.1f} saniye")
    logger.info("Model kaydedildi: models/fake_news_model.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fake News Detection - Model Eğitim Scripti'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data',
        help='Veri dizini veya CSV dosya yolu (varsayılan: ./data)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Test seti oranı (varsayılan: 0.2)'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Rastgele tohum (varsayılan: 42)'
    )

    args = parser.parse_args()
    train_model(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state
    )
