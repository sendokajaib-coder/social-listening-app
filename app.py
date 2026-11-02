import os
import streamlit as st
from dotenv import load_dotenv
from apify_client import ApifyClient
from openai import OpenAI
import datetime
import pandas as pd
import json

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Konfigurasi Halaman Web Streamlit
st.set_page_config(
    page_title="Social Media Listening RAG",
    page_icon="📊",
    layout="wide"
)

# Folder penyimpanan riwayat laporan
SAVE_DIR = "saved_reports"
os.makedirs(SAVE_DIR, exist_ok=True)

st.title("📊 Social Media Listening & Sentiment Assistant")
st.markdown(
    "Aplikasi AI untuk melakukan *crawling* otomatis lintas platform (**Facebook, TikTok, X, & Instagram**) via **Apify** dan merangkum intinya menggunakan **OpenAI GPT**.")

# Sidebar untuk Navigasi & Pengaturan
with st.sidebar:
    st.header("📌 Navigasi Menu")
    app_mode = st.radio("Pilih Mode", ["Crawl & Analisis Baru", "Viewer Riwayat Laporan"])

    st.markdown("---")
    st.header("⚙️ Pengaturan & Kredensial")

    default_apify = os.getenv("APIFY_API_KEY", "")
    default_openai = os.getenv("OPENAI_API_KEY", "")

    apify_key_input = st.text_input("Apify API Key", value=default_apify, type="password")
    openai_key_input = st.text_input("OpenAI API Key", value=default_openai, type="password")

    st.markdown("---")
    max_posts_per_platform = st.slider("Maksimal Postingan per Platform", min_value=3, max_value=1000, value=5, step=1)
    start_date = st.date_input(label="Pilih Tanggal Mulai (Start Date)", value=datetime.date.today())

# ==========================================
# MODE 1: CRAWL & ANALISIS BARU
# ==========================================
if app_mode == "Crawl & Analisis Baru":
    keyword_query = st.text_input("Masukkan Kata Kunci Pencarian Umum:",
                                  placeholder="Contoh: Kuliner Maluku / Pilkada Maluku")

    if st.button("Crawl & Analisis", type="primary"):
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

            # Daftar platform dan konfigurasi actor
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
                    "actor_id": "apify/instagram-search-scraper",
                    "input": {
                        "keywords": [keyword_query],
                        "maxResultsPerKeyword": max_posts_per_platform,
                        "searchType": "popular",
                        "search": keyword_query
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

                            # Ekstraksi teks aman lintas platform (termasuk penanganan struktur TikTok)
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
                                posts_text_collection.append(
                                    f"- [{source}] Postingan {index + 1}: {text_content[:400]}")

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

                        # Simpan hasil ke file JSON otomatis agar bisa dibuka di Viewer
                        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        safe_query = "".join(c if c.isalnum() else "_" for c in keyword_query)[:25]
                        filename = f"{timestamp_str}_{safe_query}.json"
                        filepath = os.path.join(SAVE_DIR, filename)

                        report_data = {
                            "query": keyword_query,
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "summary": summary_result,
                            "items": all_combined_items
                        }

                        with open(filepath, "w", encoding="utf-8") as f:
                            json.dump(report_data, f, ensure_ascii=False, indent=4)

                        # Menampilkan Hasil Analisis di Antarmuka Web
                        st.success(f"Analisis Berhasil Diselesaikan & Disimpan ke Riwayat ({filename})!")

                        st.markdown("### 📋 Hasil Ringkasan & Sentimen AI Lintas Platform")
                        st.markdown(summary_result)

                        # Tombol Save / Download File Laporan
                        st.markdown("---")
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                label="📥 Unduh Ringkasan AI (Markdown)",
                                data=summary_result,
                                file_name=f"ringkasan_{safe_query}.md",
                                mime="text/markdown"
                            )

                        # Menyiapkan data tabel dengan ekstraksi TikTok yang diperbarui
                        table_data = []
                        for i, post in enumerate(all_combined_items):
                            source = post.get("_source_platform", "Unknown")

                            # Ekstraksi Teks
                            text_content = (
                                    post.get("text", "") or
                                    post.get("caption", "") or
                                    post.get("full_text", "") or
                                    post.get("desc", "") or
                                    post.get("title", "") or
                                    post.get("pagemessage", "") or
                                    post.get("snippet", "")
                            )

                            # Ekstraksi URL (Khusus TikTok: webVideoUrl / url)
                            url_content = (
                                    post.get("webVideoUrl", "") or
                                    post.get("url", "") or
                                    post.get("webUrl", "") or
                                    post.get("postUrl", "") or
                                    post.get("link", "")
                            )

                            # Ekstraksi Author (Khusus TikTok: authorMeta / author)
                            author_raw = post.get("authorMeta", "") or post.get("author", "")
                            if isinstance(author_raw, dict):
                                author = author_raw.get("nickName", "") or author_raw.get("uniqueId",
                                                                                          "") or author_raw.get("name",
                                                                                                                "")
                            else:
                                author = (
                                        author_raw or
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

                        with col_dl2:
                            csv_data = df_raw.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Unduh Data Mentah (CSV)",
                                data=csv_data,
                                file_name=f"raw_data_{safe_query}.csv",
                                mime="text/csv"
                            )

                        st.markdown("### 🔍 Tabel Data Mentah Postingan (Raw Data)")
                        st.dataframe(df_raw, use_container_width=True)

                    except Exception as e:
                        st.error(f"❌ Gagal memproses data dengan OpenAI: {e}")
            else:
                st.warning("Tidak ada postingan yang ditemukan dari semua platform untuk kata kunci tersebut.")

# ==========================================
# MODE 2: VIEWER RIWAYAT LAPORAN
# ==========================================
elif app_mode == "Viewer Riwayat Laporan":
    st.header("📂 Riwayat Laporan Tersimpan")

    saved_files = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith(".json")], reverse=True)

    if not saved_files:
        st.info("Belum ada riwayat laporan yang tersimpan. Silakan lakukan pencarian di menu 'Crawl & Analisis Baru'.")
    else:
        selected_file = st.selectbox("Pilih Riwayat Berdasarkan Waktu & Kata Kunci:", saved_files)

        if selected_file:
            filepath = os.path.join(SAVE_DIR, selected_file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)

                st.markdown(f"**Kata Kunci:** `{saved_data.get('query')}`")
                st.markdown(f"**Waktu Crawling:** `{saved_data.get('timestamp')}`")
                st.markdown("---")

                st.markdown("### 📋 Ringkasan & Sentimen AI")
                st.markdown(saved_data.get("summary", ""))

                items = saved_data.get("items", [])
                if items:
                    st.markdown("---")
                    st.markdown(f"### 🔍 Tabel Data Mentah ({len(items)} Item)")

                    table_data = []
                    for i, post in enumerate(items):
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
                                post.get("webVideoUrl", "") or
                                post.get("url", "") or
                                post.get("webUrl", "") or
                                post.get("postUrl", "") or
                                post.get("link", "")
                        )
                        author_raw = post.get("authorMeta", "") or post.get("author", "")
                        if isinstance(author_raw, dict):
                            author = author_raw.get("nickName", "") or author_raw.get("uniqueId", "") or author_raw.get(
                                "name", "")
                        else:
                            author = (
                                    author_raw or
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

                    # Tombol unduh ulang dari viewer
                    csv_data = df_raw.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Unduh Ulang CSV dari Riwayat Ini",
                        data=csv_data,
                        file_name=f"export_{selected_file.replace('.json', '')}.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Gagal memuat file riwayat: {e}")