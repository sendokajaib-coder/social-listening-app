import os
import json
import datetime
from abc import ABC, abstractmethod
from apify_client import ApifyClient
from openai import OpenAI
import pandas as pd


# ==========================================
# 1. STRATEGY PATTERN UNTUK PLATFORM MEDIA SOSIAL
# ==========================================
class BasePlatformScraper(ABC):
    """Abstract Base Class untuk memastikan setiap platform memiliki kontrak yang konsisten."""

    def __init__(self, client: ApifyClient, query: str, max_posts: int, start_date: str):
        self.client = client
        self.query = query
        self.max_posts = max_posts
        self.start_date = start_date

    @abstractmethod
    def fetch_data(self) -> list:
        pass

    @staticmethod
    @abstractmethod
    def extract_text(post: dict) -> str:
        pass

    @staticmethod
    @abstractmethod
    def extract_author(post: dict) -> str:
        pass

    @staticmethod
    @abstractmethod
    def extract_url(post: dict) -> str:
        pass


class FacebookScraper(BasePlatformScraper):
    def fetch_data(self) -> list:
        run_input = {
            "query": self.query,
            "resultsCount": self.max_posts,
            "startDate": self.start_date,
        }
        run = self.client.actor("scraper_one/facebook-posts-search").call(run_input=run_input)
        return self.client.dataset(run["defaultDatasetId"]).list_items().items

    @staticmethod
    def extract_text(post: dict) -> str:
        return (
                post.get("text", "") or post.get("message", "") or
                post.get("postText", "") or post.get("story", "") or
                post.get("pagemessage", "") or post.get("snippet", "")
        )

    @staticmethod
    def extract_author(post: dict) -> str:
        author_raw = post.get("authorMeta", "") or post.get("author", "") or post.get("user", "")
        if isinstance(author_raw, dict):
            return author_raw.get("nickName", "") or author_raw.get("uniqueId", "") or author_raw.get("name", "")
        return author_raw or post.get("username", "") or post.get("profileName", "")

    @staticmethod
    def extract_url(post: dict) -> str:
        return post.get("url", "") or post.get("webUrl", "") or post.get("postUrl", "") or post.get("link", "")


class TikTokScraper(BasePlatformScraper):
    def fetch_data(self) -> list:
        run_input = {
            "searchQueries": [self.query],
            "resultsPerPage": self.max_posts,
            "shouldDownloadVideos": False,
        }
        run = self.client.actor("clockworks/tiktok-scraper").call(run_input=run_input)
        return self.client.dataset(run["defaultDatasetId"]).list_items().items

    @staticmethod
    def extract_text(post: dict) -> str:
        return post.get("text", "") or post.get("desc", "") or post.get("title", "")

    @staticmethod
    def extract_author(post: dict) -> str:
        author_raw = post.get("authorMeta", "") or post.get("author", "")
        if isinstance(author_raw, dict):
            return author_raw.get("nickName", "") or author_raw.get("uniqueId", "") or author_raw.get("name", "")
        return author_raw or post.get("username", "") or post.get("ownerUsername", "")

    @staticmethod
    def extract_url(post: dict) -> str:
        return post.get("webVideoUrl", "") or post.get("url", "") or post.get("webUrl", "") or post.get("link", "")


class XScraper(BasePlatformScraper):
    def fetch_data(self) -> list:
        run_input = {
            "query": self.query,
            "resultsCount": self.max_posts,
            "startDate": self.start_date,
        }
        run = self.client.actor("scraper_one/x-posts-search").call(run_input=run_input)
        return self.client.dataset(run["defaultDatasetId"]).list_items().items

    @staticmethod
    def extract_text(post: dict) -> str:
        return post.get("text", "") or post.get("full_text", "") or post.get("snippet", "")

    @staticmethod
    def extract_author(post: dict) -> str:
        author_raw = post.get("author", "") or post.get("user", "")
        if isinstance(author_raw, dict):
            return author_raw.get("name", "") or author_raw.get("screen_name", "")
        return author_raw or post.get("username", "") or post.get("screenName", "")

    @staticmethod
    def extract_url(post: dict) -> str:
        return post.get("url", "") or post.get("tweetUrl", "") or post.get("link", "")


class InstagramScraper(BasePlatformScraper):
    def fetch_data(self) -> list:
        run_input = {
            "keywords": [self.query],
            "maxResultsPerKeyword": self.max_posts,
            "searchType": "popular",
            "search": self.query
        }
        run = self.client.actor("apify/instagram-search-scraper").call(run_input=run_input)
        return self.client.dataset(run["defaultDatasetId"]).list_items().items

    @staticmethod
    def extract_text(post: dict) -> str:
        return post.get("caption", "") or post.get("text", "") or post.get("accessibilityCaption", "")

    @staticmethod
    def extract_author(post: dict) -> str:
        author_raw = post.get("ownerUsername", "") or post.get("ownerFullName", "") or post.get("author", "")
        if isinstance(author_raw, dict):
            return author_raw.get("username", "") or author_raw.get("fullName", "")
        return author_raw or post.get("username", "") or post.get("profileName", "")

    @staticmethod
    def extract_url(post: dict) -> str:
        return post.get("url", "") or post.get("postUrl", "") or post.get("link", "")


# Factory Helper untuk Instansiasi Platform Scraper Secara Dinamis
class ScraperFactory:
    _registry = {
        "Facebook": FacebookScraper,
        "TikTok": TikTokScraper,
        "X (Twitter)": XScraper,
        "Instagram": InstagramScraper
    }

    @classmethod
    def get_scraper(cls, name: str, client: ApifyClient, query: str, max_posts: int,
                    start_date: str) -> BasePlatformScraper:
        scraper_class = cls._registry.get(name)
        if not scraper_class:
            raise ValueError(f"Platform {name} tidak terdaftar.")
        return scraper_class(client, query, max_posts, start_date)


# ==========================================
# 2. SERVICE LAYER (OpenAI & Pipeline Orchestration)
# ==========================================
class SocialMediaService:
    def __init__(self, apify_key: str, openai_key: str):
        self.apify_client = ApifyClient(apify_key)
        self.openai_client = OpenAI(api_key=openai_key)

    def crawl_all_platforms(self, query: str, max_posts: int, start_date: str, status_callback=None) -> list:
        platforms = ["Facebook", "TikTok", "X (Twitter)", "Instagram"]
        all_items = []

        for p_name in platforms:
            if status_callback:
                status_callback(f"Menjalankan Apify Actor untuk **{p_name}**...")
            try:
                scraper = ScraperFactory.get_scraper(p_name, self.apify_client, query, max_posts, start_date)
                items = scraper.fetch_data()

                for item in items:
                    item["_source_platform"] = p_name

                all_items.extend(items)
                if status_callback:
                    status_callback(f"✅ Berhasil menarik {len(items)} item dari {p_name}.")
            except Exception as e:
                if status_callback:
                    status_callback(f"⚠️ Gagal menarik data dari {p_name}: {e}")

        return all_items

    def analyze_with_ai(self, query: str, items: list) -> str:
        posts_text_collection = []
        for index, post in enumerate(items):
            source = post.get("_source_platform", "Unknown")
            # Menggunakan ekstraksi dinamis berdasarkan factory/class jika diperlukan, atau fallback universal
            text = (
                    post.get("caption") or post.get("text") or post.get("message") or
                    post.get("full_text") or post.get("desc") or post.get("title") or ""
            )
            if text.strip():
                posts_text_collection.append(f"- [{source}] Postingan {index + 1}: {text[:400]}")

        combined_text = "\n".join(posts_text_collection[:30])

        system_prompt = (
            "Anda adalah seorang Analis Media Sosial (Social Media Intelligence Analyst). "
            "Tugas Anda adalah merangkum dan menarik kesimpulan naratif dari opini publik, "
            "tren, atau sentimen yang sedang hangat berdasarkan data postingan dari berbagai platform media sosial. "
            "Sertakan nama akun/tokoh yang menonjol beserta link-nya jika tersedia pada data, serta hitung persentase sentimen secara keseluruhan."
        )

        user_prompt = f"""
Berikut adalah daftar postingan gabungan dari berbagai platform mengenai pencarian: "{query}"

--- DATA POSTINGAN PUBLIK ---
{combined_text}
-----------------------------

Buatkan ringkasan eksekutif lintas platform yang mencakup:
1. Inti pembahasan utama dari publik di berbagai platform tersebut.
2. Sentimen umum (Positif, Negatif, atau Netral) beserta persentasenya secara menyeluruh.
3. Poin-poin temuan penting, perbandingan tren antar platform (jika ada), serta akun/tokoh menonjol beserta link-nya jika tersedia.
"""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content


# ==========================================
# 3. REPOSITORY LAYER (Storage & History Management)
# ==========================================
class ReportRepository:
    def __init__(self, save_dir: str = "saved_reports"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_report(self, query: str, summary: str, items: list) -> tuple:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c if c.isalnum() else "_" for c in query)[:25]
        filename = f"{timestamp_str}_{safe_query}.json"
        filepath = os.path.join(self.save_dir, filename)

        report_data = {
            "query": query,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "items": items
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=4)

        return filename, filepath

    def get_saved_files(self) -> list:
        if not os.path.exists(self.save_dir):
            return []
        return sorted([f for f in os.listdir(self.save_dir) if f.endswith(".json")], reverse=True)

    def load_report(self, filename: str) -> dict:
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def parse_items_to_dataframe(items: list) -> pd.DataFrame:
        table_data = []
        for i, post in enumerate(items):
            source = post.get("_source_platform", "Unknown")

            # Ekstraksi aman berdasarkan platform
            scraper = ScraperFactory._registry.get(source)
            if scraper:
                text_content = scraper.extract_text(post)
                author = scraper.extract_author(post)
                url_content = scraper.extract_url(post)
            else:
                text_content = post.get("text", "")
                author = post.get("author", "")
                url_content = post.get("url", "")

            table_data.append({
                "No": i + 1,
                "Platform": source,
                "Author / Akun": str(author),
                "Teks / Konten": text_content,
                "Link": url_content
            })

        return pd.DataFrame(table_data)