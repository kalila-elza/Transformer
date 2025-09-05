import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import io

# === Konfigurasi Halaman ===
st.set_page_config(page_title="Analisis Sentimen Tom Lembong", page_icon="📊", layout="wide")

# === Load Model (ubah path sesuai model kamu) ===
@st.cache_resource
def load_sentiment_model():
    return load_model('/content/indobert_text_classification_model')  # ganti dengan model kamu

model = load_sentiment_model()

# === Label Sentimen ===
labels = {
    0: "Sangat Negatif",
    1: "Negatif",
    2: "Netral",
    3: "Positif",
    4: "Sangat Positif"
}

# === Sidebar Navigasi ===
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "🔍 Analisis Sentimen", "ℹ Tentang"])

# ===========================
# === Halaman HOME ===
# ===========================
if page == "🏠 Home":
    st.title("📊 Analisis Sentimen Tentang Tom Lembong")
    st.image("https://pbs.twimg.com/profile_images/1638126050346874882/S8C6bEDB_400x400.jpg", caption="Tom Lembong")
    st.markdown("""
    **Siapa itu Tom Lembong?**  
    Tom Lembong (Thomas Trikasih Lembong) adalah seorang tokoh publik Indonesia, mantan Menteri Perdagangan, 
    dan Ketua BKPM (Badan Koordinasi Penanaman Modal). Beliau dikenal aktif memberikan pandangan terkait ekonomi dan politik di media sosial.

    **Tentang Aplikasi:**  
    Aplikasi ini melakukan **analisis sentimen** terhadap opini publik tentang Tom Lembong.  
    Data dikumpulkan **secara manual** dari Instagram, X (Twitter), dan Facebook.
    """)

# ===========================
# === Halaman ANALISIS ===
# ===========================
elif page == "🔍 Analisis Sentimen":
    st.title("🔍 Analisis Sentimen Komentar")
    st.markdown("""
    Masukkan komentar atau pendapat terkait **Tom Lembong**, lalu klik **Prediksi**.  
    Model ini akan mengklasifikasikan sentimen ke dalam 5 kategori:
    - **0 = Sangat Negatif**
    - **1 = Negatif**
    - **2 = Netral**
    - **3 = Positif**
    - **4 = Sangat Positif**
    """)

    user_input = st.text_area("Masukkan teks di sini:", height=150)

    if st.button("Prediksi"):
        if user_input.strip():
            # Preprocessing (contoh sederhana, sesuaikan dengan pipeline model kamu)
            # Misal tokenisasi dengan tokenizer yang sama dengan saat training
            # Untuk demo ini kita gunakan input dummy
            input_tensor = tf.constant([user_input])
            
            prediction = model.predict(input_tensor)
            pred_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.subheader(f"Hasil Prediksi: {labels[pred_index]}")
            st.write(f"Confidence: **{confidence:.2f}%**")

            st.progress(int(confidence))
        else:
            st.warning("Masukkan teks terlebih dahulu.")

# ===========================
# === Halaman ABOUT ===
# ===========================
elif page == "ℹ Tentang":
    st.title("ℹ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat untuk menganalisis sentimen komentar publik terhadap **Tom Lembong** 
    menggunakan model **Deep Learning (IndoBERT)**.  
    Dataset dikumpulkan secara manual dari **Instagram**, **X (Twitter)**, dan **Facebook**.  
    """)

    st.subheader("Label Sentimen:")
    for key, val in labels.items():
        st.write(f"- **{key}** : {val}")
