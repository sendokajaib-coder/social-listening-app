import os
import streamlit as st
from dotenv import load_dotenv
from apify_client import ApifyClient
from openai import OpenAI
import datetime
import pandas as pd

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
    "Aplikasi AI untuk melakukan *crawling* otomatis lintas platform (**Facebook, TikTok, X, & Instagram**) via **Apify** dan merangkum intinya menggunakan **OpenAI GPT**.")

# Sidebar untuk Pengaturan Kredensial & Parameter
with st.sidebar:
    st.header("⚙️ Pengaturan & Kredensial")

    default_apify = os.getenv("APIFY_API_KEY", "")
    default_openai = os.getenv("OPENAI_API_KEY", "")

    apify_key_input = st.text_input("Apify API Key", value=default_apify, type="password")
    openai_key_input = st.text_input("OpenAI API Key", value=default_openai, type="password")

    st.markdown("---")
    max_posts_per_platform = st.slider("Maksimal Postingan per Platform", min_value=3, max_value=15, value=5, step=1)
    start_date = st.date_input(label="Pilih Tanggal Mulai (Start Date)", value=datetime.date.today())

# Kolom Masukan Pencarian Pengguna
keyword_query = st.text_input("Masukkan Kata Kunci Pencarian Umum:",
                              placeholder="Contoh: Kuliner Maluku / Pilkada Maluku")

if st.button("🚀 Mulai Crawling Otomatis & Analisis", type="primary"):
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

        all_combined_items = []

        # Daftar platform dan konfigurasi actor-nya masing-masing secara otomatis
        platforms_config = [
            {
                "name": "Facebook",
                "actor_id": "scraper_one/facebook-posts-search",
                "input": {
                    "query": keyword_query,
                    "resultsCount": max_posts_per_platform,
                    "startDate": str(start_date),
                }
            },
            {
                "name": "TikTok",
                "actor_id": "clockworks/tiktok-scraper",
                "input": {
                    "searchQueries": [keyword_query],
                    "resultsPerPage": max_posts_per_platform,
                    "shouldDownloadVideos": False,
                }
            },
            {
                "name": "X (Twitter)",
                "actor_id": "scraper_one/x-posts-search",
                "input": {
                    "query": keyword_query,
                    "resultsCount": max_posts_per_platform,
                    "startDate": str(start_date),
                }
            },
            {
                "name": "Instagram",
                "actor_id": "apify/instagram-scraper",
                "input": {
                    "search": keyword_query,
                    "searchType": "hashtag",
                    "resultsLimit": max_posts_per_platform,
                }
            }
        ]

        # Proses 1: Menjalankan Crawling Otomatis ke Semua Platform
        with st.status("🔄 Sedang menjalankan crawling otomatis lintas platform...", expanded=True) as status:
            for p in platforms_config:
                p_name = p["name"]
                st.write(f"Menjalankan Apify Actor untuk **{p_name}**...")
                try:
                    run = apify_client.actor(p["actor_id"]).call(run_input=p["input"])
                    dataset_items = apify_client.dataset(run["defaultDatasetId"]).list_items().items

                    # Beri label platform pada setiap item data mentah
                    for item in dataset_items:
                        item["_source_platform"] = p_name

                    all_combined_items.extend(dataset_items)
                    st.write(f"✅ Berhasil menarik {len(dataset_items)} item dari {p_name}.")
                except Exception as e:
                    st.warning(f"⚠️ Gagal menarik data dari {p_name}: {e}")

            status.update(
                label=f"Crawling Selesai! Total {len(all_combined_items)} data terkumpul. Menganalisis dengan OpenAI...",
                state="running")

        # Proses 2: Menganalisis & Merangkum dengan OpenAI GPT
        if all_combined_items:
            with st.spinner("🤖 OpenAI sedang menyusun ringkasan eksekutif dan analisis lintas platform..."):
                try:
                    posts_text_collection = []
                    for index, post in enumerate(all_combined_items):
                        source = post.get("_source_platform", "Unknown")
                        # Ekstraksi teks dinamis sesuai struktur platform
                        text_content = (
                                post.get("text", "") or
                                post.get("caption", "") or
                                post.get("full_text", "") or
                                post.get("desc", "") or
                                post.get("title", "") or
                                post.get("pagemessage", "") or
                                post.get("snippet", "")
                        )
                        if text_content.strip():
                            posts_text_collection.append(f"- [{source}] Postingan {index + 1}: {text_content[:400]}")

                    combined_text = "\n".join(posts_text_collection[:30])

                    system_prompt = (
                        "Anda adalah seorang Analis Media Sosial (Social Media Intelligence Analyst). "
                        "Tugas Anda adalah merangkum dan menarik kesimpulan naratif dari opini publik, "
                        "tren, atau sentimen yang sedang hangat berdasarkan data postingan dari berbagai platform media sosial. "
                        "Sertakan nama akun/tokoh yang menonjol beserta link-nya jika tersedia pada data, serta hitung persentase sentimen secara keseluruhan."
                    )

                    user_prompt = f"""
Berikut adalah daftar postingan gabungan dari berbagai platform (Facebook, TikTok, X, Instagram) mengenai pencarian: "{keyword_query}"

--- DATA POSTINGAN PUBLIK ---
{combined_text}
-----------------------------

Buatkan ringkasan eksekutif lintas platform yang mencakup:
1. Inti pembahasan utama dari publik di berbagai platform tersebut.
2. Sentimen umum (Positif, Negatif, atau Netral) beserta persentasenya secara menyeluruh.
3. Poin-poin temuan penting, perbandingan tren antar platform (jika ada), serta akun/tokoh menonjol beserta link-nya jika tersedia.
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
                    st.success("Analisis Lintas Platform Berhasil Diselesaikan!")

                    st.markdown("### 📋 Hasil Ringkasan & Sentimen AI Lintas Platform")
                    st.markdown(summary_result)

                    # Menampilkan Data Mentah dalam Bentuk Tabel Interaktif
                    st.markdown("---")
                    st.markdown("### 🔍 Tabel Data Mentah Postingan (Raw Data)")

                    table_data = []
                    for i, post in enumerate(all_combined_items):
                        source = post.get("_source_platform", "Unknown")
                        text_content = (
                                post.get("text", "") or
                                post.get("caption", "") or
                                post.get("full_text", "") or
                                post.get("desc", "") or
                                post.get("title", "") or
                                post.get("pagemessage", "") or
                                post.get("snippet", "")
                        )
                        url_content = (
                                post.get("url", "") or
                                post.get("webUrl", "") or
                                post.get("postUrl", "") or
                                post.get("link", "")
                        )
                        author = (
                                post.get("author", "") or
                                post.get("username", "") or
                                post.get("ownerUsername", "") or
                                post.get("user", "")
                        )

                        table_data.append({
                            "No": i + 1,
                            "Platform": source,
                            "Author / Akun": str(author),
                            "Teks / Konten": text_content,
                            "Link": url_content
                        })

                    df_raw = pd.DataFrame(table_data)
                    st.dataframe(df_raw, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Gagal memproses data dengan OpenAI: {e}")
        else:
            st.warning("Tidak ada postingan yang ditemukan dari semua platform untuk kata kunci tersebut.")