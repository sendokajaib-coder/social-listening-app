import os
import streamlit as st
from dotenv import load_dotenv
from apify_client import ApifyClient
from openai import OpenAI
import datetime

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Konfigurasi Halaman Web Streamlit
st.set_page_config(
    page_title="Social Media Listening RAG",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Social Media Listening & Sentiment Assistant")
st.markdown(
    "Aplikasi AI untuk melakukan *crawling* postingan Facebook via **Apify** dan merangkum intinya menggunakan **OpenAI GPT**.")

# Sidebar untuk Pengaturan Kredensial & Parameter
with st.sidebar:
    st.header("⚙️ Pengaturan & Kredensial")

    default_apify = os.getenv("APIFY_API_KEY", "")
    default_openai = os.getenv("OPENAI_API_KEY", "")

    apify_key_input = st.text_input("Apify API Key", value=default_apify, type="password")
    openai_key_input = st.text_input("OpenAI API Key", value=default_openai, type="password")

    st.markdown("---")
    max_posts = st.slider("Jumlah Maksimal Postingan", min_value=5, max_value=30, value=10, step=5)
    start_date = start_date = st.date_input(label="Pilih Tanggal Mulai (Start Date)",value=datetime.date.today())

# Kolom Masukan Pencarian Pengguna
keyword_query = st.text_input("Masukkan Kata Kunci untuk Pencarian Facebook:", placeholder="Contoh: Kuliner Maluku")

if st.button("🚀 Mulai Crawling & Analisis", type="primary"):
    if not apify_key_input or not openai_key_input:
        st.error("⚠️ Mohon masukkan Apify API Key dan OpenAI API Key terlebih dahulu di sidebar atau file .env!")
    elif not keyword_query.strip():
        st.warning("⚠️ Kata kunci tidak boleh kosong!")
    else:
        try:
            apify_client = ApifyClient(apify_key_input)
            openai_client = OpenAI(api_key=openai_key_input)
        except Exception as e:
            st.error(f"❌ Gagal menginisialisasi API Client: {e}")
            st.stop()

        # Proses 1: Mengambil Data dari Facebook via Apify Actor
        with st.status("🔄 Sedang berjalan...", expanded=True) as status:
            st.write(f"Menjalankan Apify Actor untuk query: **{keyword_query}**...")

            try:
                # Menggunakan actor spesifik sesuai permintaan
                actor_id = "scraper_one/facebook-posts-search"

                run_input = {
                    "query": keyword_query,
                    "resultsCount": max_posts,
                    "startDate": start_date,
                }

                run = apify_client.actor(actor_id).call(run_input=run_input)
                dataset_items = apify_client.dataset(run["defaultDatasetId"]).list_items().items

                st.write(f"✅ Berhasil menarik {len(dataset_items)} postingan dari Facebook.")
                status.update(label="Crawling Selesai! Menganalisis dengan OpenAI...", state="running")

            except Exception as e:
                st.error(f"❌ Gagal melakukan crawling: {e}")
                dataset_items = []
                status.update(label="Terjadi Kesalahan", state="error")

        # Proses 2: Menganalisis & Merangkum dengan OpenAI GPT
        if dataset_items:
            with st.spinner("🤖 OpenAI sedang menyusun ringkasan eksekutif dan analisis sentimen..."):
                try:
                    posts_text_collection = []
                    for index, post in enumerate(dataset_items):
                        text_content = post.get("text", "") or post.get("caption", "") or post.get("pagemessage",
                                                                                                   "") or post.get(
                            "snippet", "")
                        if text_content.strip():
                            posts_text_collection.append(f"- Postingan {index + 1}: {text_content[:500]}")

                    combined_text = "\n".join(posts_text_collection[:15])

                    system_prompt = (
                        "Anda adalah seorang Analis Media Sosial (Social Media Intelligence Analyst). "
                        "Tugas Anda adalah merangkum dan menarik kesimpulan naratif dari opini publik, "
                        "tren, atau sentimen yang sedang hangat berdasarkan data postingan media sosial."
                    )

                    user_prompt = f"""
Berikut adalah daftar postingan Facebook mengenai pencarian: "{keyword_query}"

--- DATA POSTINGAN PUBLIK ---
{combined_text}
-----------------------------

Buatkan ringkasan eksekutif yang mencakup:
1. Inti pembahasan utama dari publik.
2. Sentimen umum (Positif, Negatif, atau Netral).
3. Poin-poin temuan penting atau isu yang menonjol.
"""

                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.4
                    )

                    summary_result = response.choices[0].message.content

                    # Menampilkan Hasil Analisis di Antarmuka Web
                    st.success("Analisis Berhasil Diselesaikan!")

                    st.markdown("### 📋 Hasil Ringkasan & Sentimen AI")
                    st.markdown(summary_result)

                    # Menampilkan Data Mentah dalam Bentuk Dropdown (Expandable)
                    with st.expander("🔍 Lihat Data Mentah Postingan (Raw Data)"):
                        for i, post in enumerate(dataset_items):
                            st.write(f"**Postingan {i + 1}**")
                            st.json(post)

                except Exception as e:
                    st.error(f"❌ Gagal memproses data dengan OpenAI: {e}")
        else:
            st.warning("Tidak ada postingan yang ditemukan untuk kata kunci tersebut.")