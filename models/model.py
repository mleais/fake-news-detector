"""
Fake News Detection Model Module
==================================
TF-IDF + Logistic Regression tabanlı sahte haber tespit modeli.
"""

import os
import pickle
import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

logger = logging.getLogger(__name__)

# Model kayıt yolu
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fake_news_model.pkl')


class FakeNewsDetector:
    """
    TF-IDF ve Logistic Regression tabanlı sahte haber tespit modeli.

    Bu sınıf aşağıdaki adımları gerçekleştirir:
    - TF-IDF vektörizasyonu ile metin sayısallaştırma
    - Logistic Regression ile binary sınıflandırma
    - Model kaydetme ve yükleme
    - Tahmin sonuçlarını olasılık ile döndürme

    Örnek Kullanım:
        >>> detector = FakeNewsDetector()
        >>> detector.train(X_train, y_train)
        >>> result = detector.predict("Breaking news: scientists discover...")
        >>> print(result)
        {'label': 'REAL', 'confidence': 0.87, 'fake_prob': 0.13, 'real_prob': 0.87}
    """

    # Sınıf sabitleri
    FAKE_LABEL = "FAKE"
    REAL_LABEL = "REAL"
    LABEL_MAP = {0: REAL_LABEL, 1: FAKE_LABEL}
    REVERSE_LABEL_MAP = {REAL_LABEL: 0, FAKE_LABEL: 1}

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 2),
        max_iter: int = 1000,
        C: float = 1.0
    ) -> None:
        """
        FakeNewsDetector başlatıcısı.

        Args:
            max_features: TF-IDF maksimum özellik sayısı
            ngram_range: N-gram aralığı (unigram + bigram)
            max_iter: Logistic Regression maksimum iterasyon
            C: Regularization parametresi
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_iter = max_iter
        self.C = C
        self.is_trained = False
        self.pipeline: Optional[Pipeline] = None
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """
        Sklearn Pipeline oluşturur.
        TF-IDF → Logistic Regression
        """
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=True,       # log(1 + tf) dönüşümü
                min_df=2,                 # En az 2 belgede geçen terimler
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\b[a-zA-Z]{2,}\b'
            )),
            ('classifier', LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                solver='lbfgs',
                class_weight='balanced',  # Dengesiz veri setleri için
                n_jobs=-1
            ))
        ])

    def train(self, X_train: list, y_train: list) -> None:
        """
        Modeli eğitir.

        Args:
            X_train: Eğitim metinleri listesi
            y_train: Etiket listesi (0: gerçek, 1: sahte)
        """
        logger.info(f"Model eğitimi başlıyor... {len(X_train)} örnek")
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Model eğitimi tamamlandı.")

    def predict(self, text: str) -> Dict:
        """
        Tek metin için tahmin yapar.

        Args:
            text: Analiz edilecek haber metni

        Returns:
            Tahmin sonucu dict:
                - label: 'FAKE' veya 'REAL'
                - confidence: En yüksek olasılık (0-1)
                - fake_prob: Sahte haber olasılığı
                - real_prob: Gerçek haber olasılığı

        Raises:
            RuntimeError: Model eğitilmemişse
        """
        if not self.is_trained:
            raise RuntimeError("Model henüz eğitilmemiş. Önce train() çağırın veya model yükleyin.")

        proba = self.pipeline.predict_proba([text])[0]
        pred_class = self.pipeline.predict([text])[0]

        # Sınıf indeksleri: pipeline'daki sıraya göre alınır
        classes = self.pipeline.classes_
        class_to_idx = {c: i for i, c in enumerate(classes)}

        real_idx = class_to_idx.get(0, 0)
        fake_idx = class_to_idx.get(1, 1)

        real_prob = float(proba[real_idx])
        fake_prob = float(proba[fake_idx])
        label = self.LABEL_MAP.get(int(pred_class), "UNKNOWN")
        confidence = max(real_prob, fake_prob)

        return {
            'label': label,
            'confidence': round(confidence, 4),
            'fake_prob': round(fake_prob, 4),
            'real_prob': round(real_prob, 4)
        }

    def evaluate(self, X_test: list, y_test: list) -> Dict:
        """
        Model performansını değerlendirir.

        Args:
            X_test: Test metinleri
            y_test: Gerçek etiketler

        Returns:
            Metrik sözlüğü: accuracy, precision, recall, f1
        """
        if not self.is_trained:
            raise RuntimeError("Model henüz eğitilmemiş.")

        y_pred = self.pipeline.predict(X_test)

        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
            'recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
            'f1_score': round(f1_score(y_test, y_pred, average='weighted'), 4),
            'report': classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        logger.info(f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")
        return metrics

    def save(self, path: str = MODEL_PATH) -> None:
        """
        Modeli diske kaydeder.

        Args:
            path: Kayıt yolu
        """
        if not self.is_trained:
            raise RuntimeError("Kaydedilecek eğitilmiş model yok.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'is_trained': self.is_trained,
                'params': {
                    'max_features': self.max_features,
                    'ngram_range': self.ngram_range,
                    'max_iter': self.max_iter,
                    'C': self.C
                }
            }, f)
        logger.info(f"Model kaydedildi: {path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> 'FakeNewsDetector':
        """
        Kaydedilmiş modeli yükler.

        Args:
            path: Model dosyası yolu

        Returns:
            Yüklenmiş FakeNewsDetector nesnesi

        Raises:
            FileNotFoundError: Model dosyası bulunamazsa
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        params = data.get('params', {})
        detector = cls(**params)
        detector.pipeline = data['pipeline']
        detector.is_trained = data['is_trained']
        logger.info(f"Model yüklendi: {path}")
        return detector

    @staticmethod
    def is_model_available(path: str = MODEL_PATH) -> bool:
        """
        Model dosyasının mevcut olup olmadığını kontrol eder.

        Args:
            path: Model dosyası yolu

        Returns:
            Dosya mevcutsa True
        """
        return os.path.exists(path)
