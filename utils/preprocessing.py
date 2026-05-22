"""
Text Preprocessing Module for Fake News Detection
===================================================
Bu modül, metin verilerini makine öğrenmesi modeli için hazırlar.
"""

import re
import string
import logging
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# NLTK kaynaklarını indir
def download_nltk_resources() -> None:
    """Gerekli NLTK kaynaklarını indirir."""
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            logging.warning(f"NLTK resource '{resource}' indirilemedi: {e}")

download_nltk_resources()

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Metin ön işleme sınıfı.

    Bu sınıf haber metinleri üzerinde aşağıdaki adımları uygular:
    - Küçük harfe çevirme
    - URL ve e-posta temizleme
    - Noktalama işareti temizleme
    - Sayı temizleme
    - Stopword kaldırma
    - Stemming veya lemmatization

    Örnek Kullanım:
        >>> preprocessor = TextPreprocessor(use_lemmatization=True)
        >>> clean_text = preprocessor.preprocess("This is a FAKE news article!!!")
        >>> print(clean_text)
        'fake news article'
    """

    def __init__(
        self,
        use_stemming: bool = False,
        use_lemmatization: bool = True,
        remove_stopwords: bool = True,
        language: str = "english"
    ) -> None:
        """
        TextPreprocessor başlatıcısı.

        Args:
            use_stemming: Stemming uygulansın mı?
            use_lemmatization: Lemmatization uygulansın mı?
            remove_stopwords: Stopword'ler kaldırılsın mı?
            language: Stopword dili (varsayılan: İngilizce)
        """
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.language = language

        # Araçları başlat
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None

        try:
            self.stop_words = set(stopwords.words(language))
        except OSError:
            logger.warning("Stopword listesi yüklenemedi. Boş liste kullanılıyor.")
            self.stop_words = set()

    def _to_lowercase(self, text: str) -> str:
        """Metni küçük harfe çevirir. O(n)"""
        return text.lower()

    def _remove_urls(self, text: str) -> str:
        """URL ve e-posta adreslerini temizler. O(n)"""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        return text

    def _remove_punctuation(self, text: str) -> str:
        """Noktalama işaretlerini kaldırır. O(n)"""
        return text.translate(str.maketrans('', '', string.punctuation))

    def _remove_numbers(self, text: str) -> str:
        """Sayıları kaldırır. O(n)"""
        return re.sub(r'\d+', '', text)

    def _remove_extra_whitespace(self, text: str) -> str:
        """Fazla boşlukları temizler. O(n)"""
        return ' '.join(text.split())

    def _remove_special_chars(self, text: str) -> str:
        """Özel karakterleri temizler. O(n)"""
        return re.sub(r'[^\w\s]', '', text)

    def _tokenize(self, text: str) -> List[str]:
        """Metni tokenlara ayırır. O(n)"""
        try:
            return word_tokenize(text)
        except Exception:
            return text.split()

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Stopword'leri kaldırır. O(n)"""
        return [token for token in tokens if token not in self.stop_words]

    def _apply_stemming(self, tokens: List[str]) -> List[str]:
        """Stemming uygular. O(n*k) - k: ortalama token uzunluğu"""
        if self.stemmer is None:
            return tokens
        return [self.stemmer.stem(token) for token in tokens]

    def _apply_lemmatization(self, tokens: List[str]) -> List[str]:
        """Lemmatization uygular. O(n*k)"""
        if self.lemmatizer is None:
            return tokens
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess(self, text: str) -> str:
        """
        Tam metin ön işleme pipeline'ı.

        Sıra önemlidir: lowercase → URL temizle → noktalama → 
        sayılar → tokenize → stopword → stem/lemma → birleştir

        Args:
            text: Ham haber metni

        Returns:
            Temizlenmiş ve normalleştirilmiş metin

        Time Complexity: O(n) - n: metin uzunluğu
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Adım 1: Küçük harfe çevir
        text = self._to_lowercase(text)

        # Adım 2: URL temizle
        text = self._remove_urls(text)

        # Adım 3: Noktalama kaldır
        text = self._remove_punctuation(text)

        # Adım 4: Sayıları kaldır
        text = self._remove_numbers(text)

        # Adım 5: Özel karakterleri kaldır
        text = self._remove_special_chars(text)

        # Adım 6: Tokenize et
        tokens = self._tokenize(text)

        # Adım 7: Stopword kaldır
        if self.remove_stopwords:
            tokens = self._remove_stopwords(tokens)

        # Adım 8: Kısa tokenleri filtrele (1 karakter)
        tokens = [t for t in tokens if len(t) > 1]

        # Adım 9: Stemming veya Lemmatization
        if self.use_stemming:
            tokens = self._apply_stemming(tokens)
        elif self.use_lemmatization:
            tokens = self._apply_lemmatization(tokens)

        # Adım 10: Tekrar birleştir ve fazla boşlukları temizle
        text = self._remove_extra_whitespace(' '.join(tokens))

        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Toplu metin ön işleme.

        Args:
            texts: Ham metin listesi

        Returns:
            Temizlenmiş metin listesi

        Time Complexity: O(n*m) - n: metin sayısı, m: ortalama uzunluk
        """
        return [self.preprocess(text) for text in texts]
