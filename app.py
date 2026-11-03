import os
import streamlit as st
from dotenv import load_dotenv
import datetime
from social_services import SocialMediaService, ReportRepository

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Konfigurasi Halaman Web Streamlit
st.set_page_config(
    page_title="Social Media Listening RAG",
    page_icon="📊",
    layout="wide"
)

# Inisialisasi Repository
repository = ReportRepository("saved_reports")

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
                service = SocialMediaService(apify_key_input, openai_key_input)
            except Exception as e:
                st.error(f"❌ Gagal menginisialisasi Service: {e}")
                st.stop()

            # Proses 1: Crawling Lintas Platform
            status_box = st.status("🔄 Sedang menjalankan crawling otomatis lintas platform...", expanded=True)
            with status_box:
                all_combined_items = service.crawl_all_platforms(
                    query=keyword_query,
                    max_posts=max_posts_per_platform,
                    start_date=str(start_date),
                    status_callback=st.write
                )
                status_box.update(
                    label=f"Crawling Selesai! Total {len(all_combined_items)} data terkumpul. Menganalisis dengan OpenAI...",
                    state="running"
                )

            # Proses 2: Analisis AI & Penyimpanan
            if all_combined_items:
                with st.spinner("🤖 OpenAI sedang menyusun ringkasan eksekutif dan analisis lintas platform..."):
                    try:
                        summary_result = service.analyze_with_ai(keyword_query, all_combined_items)
                        filename, filepath = repository.save_report(keyword_query, summary_result, all_combined_items)

                        st.success(f"Analisis Berhasil Diselesaikan & Disimpan ke Riwayat ({filename})!")

                        st.markdown("### 📋 Hasil Ringkasan & Sentimen AI Lintas Platform")
                        st.markdown(summary_result)

                        # Tombol Unduh Laporan
                        st.markdown("---")
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            safe_query = "".join(c if c.isalnum() else "_" for c in keyword_query)[:25]
                            st.download_button(
                                label="📥 Unduh Ringkasan AI (Markdown)",
                                data=summary_result,
                                file_name=f"ringkasan_{safe_query}.md",
                                mime="text/markdown"
                            )

                        df_raw = repository.parse_items_to_dataframe(all_combined_items)

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

    saved_files = repository.get_saved_files()

    if not saved_files:
        st.info("Belum ada riwayat laporan yang tersimpan. Silakan lakukan pencarian di menu 'Crawl & Analisis Baru'.")
    else:
        selected_file = st.selectbox("Pilih Riwayat Berdasarkan Waktu & Kata Kunci:", saved_files)

        if selected_file:
            try:
                saved_data = repository.load_report(selected_file)

                st.markdown(f"**Kata Kunci:** `{saved_data.get('query')}`")
                st.markdown(f"**Waktu Crawling:** `{saved_data.get('timestamp')}`")
                st.markdown("---")

                st.markdown("### 📋 Ringkasan & Sentimen AI")
                st.markdown(saved_data.get("summary", ""))

                items = saved_data.get("items", [])
                if items:
                    st.markdown("---")
                    st.markdown(f"### 🔍 Tabel Data Mentah ({len(items)} Item)")

                    df_raw = repository.parse_items_to_dataframe(items)
                    st.dataframe(df_raw, use_container_width=True)

                    csv_data = df_raw.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Unduh Ulang CSV dari Riwayat Ini",
                        data=csv_data,
                        file_name=f"export_{selected_file.replace('.json', '')}.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Gagal memuat file riwayat: {e}")