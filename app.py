"""
Fake News Detection - Streamlit Web Arayüzü
============================================
Kullanıcı dostu web arayüzü ile haber metni analizi.

Çalıştırma:
    streamlit run app.py
"""

import os
import sys
import time
import logging

import streamlit as st

# Modül yolları
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import TextPreprocessor
from models.model import FakeNewsDetector, MODEL_PATH

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Sayfa Konfigürasyonu
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CSS Stilleri
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Ana başlık */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    /* Sonuç kartları */
    .result-fake {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        box-shadow: 0 4px 20px rgba(255, 65, 108, 0.4);
        animation: pulse 1.5s ease-in-out;
    }
    .result-real {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        box-shadow: 0 4px 20px rgba(56, 239, 125, 0.4);
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #0f3460;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.4rem 0;
    }
    @keyframes pulse {
        0% { transform: scale(0.97); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1.0); }
    }
    /* Buton */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0f3460, #533483);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.7rem;
        border-radius: 8px;
        border: none;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
    }
    /* Sidebar */
    .sidebar-info {
        background: #eef2ff;
        padding: 0.8rem;
        border-radius: 8px;
        font-size: 0.85rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Model Yükleme (Cache)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_cached():
    """Modeli yükler ve önbellekte tutar."""
    if FakeNewsDetector.is_model_available():
        try:
            return FakeNewsDetector.load(), True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            return None, False
    return None, False


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    """TextPreprocessor'ı yükler."""
    return TextPreprocessor(use_lemmatization=True, remove_stopwords=True)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/news.png", width=64)
    st.title("Hakkında")

    st.markdown("""
    <div class='sidebar-info'>
    Bu uygulama, haber metinlerini analiz ederek <b>gerçek mi yoksa sahte mi</b> olduğunu tahmin eder.
    <br><br>
    <b>🛠 Teknoloji:</b><br>
    • TF-IDF Vektörizasyon<br>
    • Logistic Regression<br>
    • NLTK Metin İşleme<br><br>
    <b>📊 Eğitim Verisi:</b><br>
    • Fake and Real News Dataset<br>
    • ~44,000 haber makalesi
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**⚙️ Ayarlar**")
    show_details = st.checkbox("Detaylı analiz göster", value=True)
    show_preprocessing = st.checkbox("Ön işleme sonucunu göster", value=False)

    st.divider()
    st.caption("v1.0.0 | Yapay Zeka Destekli Sahte Haber Tespit Sistemi")


# ──────────────────────────────────────────────
# Ana Sayfa
# ──────────────────────────────────────────────
st.markdown("<h1 class='main-title'>🔍 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Yapay Zeka Destekli Sahte Haber Tespit Sistemi</p>", unsafe_allow_html=True)

# Model yükleme durumunu kontrol et
detector, model_loaded = load_model_cached()
preprocessor = load_preprocessor()

if not model_loaded:
    st.warning("""
    ⚠️ **Eğitilmiş model bulunamadı.**
    
    Modeli eğitmek için:
    1. Kaggle'dan veri setini indirin: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
    2. `Fake.csv` ve `True.csv` dosyalarını `data/` klasörüne kopyalayın
    3. Terminalde çalıştırın: `python train.py`
    
    **Demo modu:** Model olmadan da metin analizi deneyebilirsiniz (sonuçlar simüle edilmiştir).
    """)

# ──────────────────────────────────────────────
# Giriş Seçimi
# ──────────────────────────────────────────────
tab1, tab2 = st.tabs(["✏️ Metin Gir", "📄 Dosya Yükle"])

news_text = ""

with tab1:
    news_text_input = st.text_area(
        "Haber metnini buraya yapıştırın:",
        height=200,
        placeholder="Örnek: Scientists have discovered a new method to generate clean energy using solar panels...",
        key="text_input"
    )
    if news_text_input:
        news_text = news_text_input

    # Örnek haberler
    st.markdown("**📌 Örnek Haberler:**")
    col1, col2 = st.columns(2)

    SAMPLE_REAL = (
        "NASA's Perseverance rover successfully collected rock samples from Mars that scientists "
        "believe could contain evidence of ancient microbial life. The samples will be returned "
        "to Earth by a future mission for detailed analysis in laboratories."
    )
    SAMPLE_FAKE = (
        "BREAKING: Government secretly putting mind-control chemicals in drinking water to control "
        "citizens' thoughts. Whistleblower exposes shocking truth that mainstream media refuses to "
        "report. Share before they delete this!"
    )

    if col1.button("📰 Gerçek Haber Örneği"):
        st.session_state['sample_text'] = SAMPLE_REAL
        st.rerun()

    if col2.button("🚫 Sahte Haber Örneği"):
        st.session_state['sample_text'] = SAMPLE_FAKE
        st.rerun()

    if 'sample_text' in st.session_state:
        news_text = st.session_state['sample_text']
        st.info(f"**Örnek metin yüklendi:** {news_text[:100]}...")

with tab2:
    uploaded_file = st.file_uploader(
        "Metin dosyası yükleyin (.txt)",
        type=['txt'],
        help="Analiz edilecek haber metnini içeren .txt dosyasını yükleyin"
    )
    if uploaded_file:
        try:
            news_text = uploaded_file.read().decode('utf-8')
            st.success(f"✅ Dosya yüklendi: **{uploaded_file.name}** ({len(news_text)} karakter)")
            with st.expander("Dosya içeriğini göster"):
                st.text(news_text[:1000] + ("..." if len(news_text) > 1000 else ""))
        except UnicodeDecodeError:
            st.error("❌ Dosya okunamadı. Lütfen UTF-8 kodlu bir .txt dosyası yükleyin.")

# ──────────────────────────────────────────────
# Analiz Butonu
# ──────────────────────────────────────────────
st.divider()
analyze_btn = st.button("🔍 Analiz Et", use_container_width=True, type="primary")

if analyze_btn:
    if not news_text or not news_text.strip():
        st.error("❌ Lütfen analiz edilecek bir metin girin veya dosya yükleyin.")
    elif len(news_text.split()) < 5:
        st.warning("⚠️ Daha güvenilir sonuç için en az 5 kelimelik metin girin.")
    else:
        with st.spinner("🔄 Metin analiz ediliyor..."):
            time.sleep(0.5)  # Kullanıcı deneyimi için küçük gecikme

            # Ön işleme
            processed = preprocessor.preprocess(news_text)

            if show_preprocessing:
                with st.expander("🔧 Ön İşleme Sonucu"):
                    st.text(processed if processed else "(Metin ön işlemeden sonra boş kaldı)")

            # Tahmin
            if model_loaded and detector:
                try:
                    result = detector.predict(processed if processed else news_text)
                except Exception as e:
                    st.error(f"Tahmin hatası: {e}")
                    st.stop()
            else:
                # Demo modu: basit kural tabanlı
                fake_keywords = ['breaking', 'shocking', 'secret', 'hoax', 'conspiracy',
                                 'they dont want', 'share before', 'whistleblower', 'exposed']
                text_lower = news_text.lower()
                fake_score = sum(1 for kw in fake_keywords if kw in text_lower)
                fake_prob = min(0.4 + fake_score * 0.1, 0.95)
                real_prob = 1 - fake_prob
                label = "FAKE" if fake_prob > 0.5 else "REAL"
                result = {
                    'label': label,
                    'confidence': round(max(fake_prob, real_prob), 4),
                    'fake_prob': round(fake_prob, 4),
                    'real_prob': round(real_prob, 4)
                }

        # ── Sonuçları Göster ──
        st.divider()
        st.markdown("### 📊 Analiz Sonucu")

        if result['label'] == 'FAKE':
            st.markdown(f"""
            <div class='result-fake'>
                🚨 SAHTE HABER (FAKE NEWS)<br>
                <span style='font-size:1rem;opacity:0.9'>Güven: %{result['confidence']*100:.1f}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-real'>
                ✅ GERÇEK HABER (REAL NEWS)<br>
                <span style='font-size:1rem;opacity:0.9'>Güven: %{result['confidence']*100:.1f}</span>
            </div>
            """, unsafe_allow_html=True)

        if show_details:
            st.markdown("")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="🔴 Sahte Haber İhtimali",
                    value=f"%{result['fake_prob']*100:.1f}"
                )
            with col2:
                st.metric(
                    label="🟢 Gerçek Haber İhtimali",
                    value=f"%{result['real_prob']*100:.1f}"
                )
            with col3:
                st.metric(
                    label="🎯 Model Güveni",
                    value=f"%{result['confidence']*100:.1f}"
                )

            # Olasılık bar grafiği
            st.markdown("**Olasılık Dağılımı:**")
            st.progress(result['fake_prob'], text=f"🔴 Sahte: %{result['fake_prob']*100:.1f}")
            st.progress(result['real_prob'], text=f"🟢 Gerçek: %{result['real_prob']*100:.1f}")

        if not model_loaded:
            st.caption("⚠️ *Demo modu: Sonuçlar eğitilmiş model olmadan tahmini değerlerdir.*")

        # Uyarı notu
        st.info(
            "ℹ️ **Önemli Not:** Bu sistem bir yapay zeka modelidir ve hata yapabilir. "
            "Haberleri doğrulamak için her zaman güvenilir haber kaynaklarını kontrol edin."
        )
