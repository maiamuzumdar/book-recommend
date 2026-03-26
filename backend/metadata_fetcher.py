"""
Fetch book metadata (descriptions, subjects, covers) from the Open Library API.

Uses a two-pass strategy:
  Pass 1: Resolve each book to an Open Library Work ID
    - Books with ISBN  → /isbn/{isbn}.json → extract works[0].key
    - Books without ISBN → /search.json?title=X&author=Y → extract docs[0].key
  Pass 2: Fetch full Work data for each book
    - /works/{work_id}.json → extract description, subjects, covers

Results are cached to data/enriched_books.json to avoid repeated API calls.
"""

import json
import os
import re
import time
from typing import Optional

import pandas as pd
import requests

# Paths
DATA_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
CACHE_PATH: str = os.path.join(DATA_DIR, "enriched_books.json")

# Open Library API config
BASE_URL: str = "https://openlibrary.org"
USER_AGENT: str = "bookrecommend/1.0 maia.muzumdar@gmail.com"
REQUEST_DELAY: float = 0.5  # ~3 requests/second with User-Agent header


class MetadataFetcher:
    """Fetches and caches book metadata from the Open Library API."""

    def __init__(self, cache_path: str = CACHE_PATH) -> None:
        self.cache_path: str = cache_path
        self.session: requests.Session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.cache: dict[str, dict] = self._load_cache()

    def _load_cache(self) -> dict[str, dict]:
        """Load cached metadata from JSON file if it exists."""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        """Save current metadata cache to JSON file."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _request(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """Make a rate-limited GET request to the Open Library API."""
        time.sleep(REQUEST_DELAY)
        try:
            response: requests.Response = self.session.get(
                url, params=params, timeout=15
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited — wait and retry once
                print(f"  Rate limited, waiting 5s...")
                time.sleep(5)
                response = self.session.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    return response.json()
            else:
                print(f"  HTTP {response.status_code} for {url}")
                return None
        except requests.RequestException as e:
            print(f"  Request error for {url}: {e}")
            return None

    def _extract_description(self, data: dict) -> str:
        """
        Extract description from an Open Library response.
        Handles the polymorphic description field which can be a plain
        string or an object like {"type": "/type/text", "value": "..."}.
        """
        desc: object = data.get("description", "")
        if isinstance(desc, dict):
            return desc.get("value", "")
        return str(desc)

    def _resolve_work_id_by_isbn(self, isbn: str) -> Optional[str]:
        """Resolve an ISBN to an Open Library Work ID via /isbn/{isbn}.json."""
        data: Optional[dict] = self._request(f"{BASE_URL}/isbn/{isbn}.json")
        if data and "works" in data and len(data["works"]) > 0:
            return data["works"][0]["key"]  # e.g., "/works/OL12345W"
        return None

    def _clean_title_for_search(self, title: str) -> str:
        """
        Strip series info, subtitles, and other noise from titles
        for better Open Library search results.
        e.g., "The Testaments (The Handmaid's Tale, #2)" → "The Testaments"
              "Atomic Habits: An Easy & Proven Way to..." → "Atomic Habits"
        """
        cleaned: str = re.sub(r"\s*\(.*?\)\s*", " ", title).strip()
        # Strip subtitles after colon — these often confuse the search
        if ":" in cleaned:
            cleaned = cleaned.split(":")[0].strip()
        cleaned = cleaned.rstrip(": ")
        return cleaned if cleaned else title

    def _resolve_work_id_by_search(
        self, title: str, author: str
    ) -> Optional[str]:
        """
        Resolve a title+author to an Open Library Work ID via Search API.
        Tries the original title first, then a cleaned version with
        series info stripped.
        """
        params: dict = {
            "title": title,
            "author": author,
            "fields": "key,title,author_name,subject,cover_i,first_publish_year",
            "limit": 1,
        }
        data: Optional[dict] = self._request(f"{BASE_URL}/search.json", params=params)
        if data and data.get("docs") and len(data["docs"]) > 0:
            return data["docs"][0].get("key")

        # Retry with cleaned title (strip series info like "(Series Name, #2)")
        cleaned_title: str = self._clean_title_for_search(title)
        if cleaned_title != title:
            params["title"] = cleaned_title
            data = self._request(f"{BASE_URL}/search.json", params=params)
            if data and data.get("docs") and len(data["docs"]) > 0:
                return data["docs"][0].get("key")

        return None

    def _fetch_work_data(self, work_key: str) -> Optional[dict]:
        """Fetch full Work data from /works/{work_id}.json."""
        return self._request(f"{BASE_URL}{work_key}.json")

    def _build_cover_url(self, cover_id: int, size: str = "M") -> str:
        """Build a cover image URL from a cover ID. No API call needed."""
        return f"https://covers.openlibrary.org/b/id/{cover_id}-{size}.jpg"

    def fetch_metadata_for_book(
        self, title: str, author: str, isbn: str, isbn13: str
    ) -> dict:
        """
        Fetch metadata for a single book. Returns a dict with:
          - work_key: Open Library Work ID
          - description: book description text
          - subjects: list of subject/genre tags
          - subject_places: list of geographic settings
          - subject_people: list of named characters/people
          - subject_times: list of time periods
          - cover_url: URL to cover image (medium size)
        """
        # Pass 1: Resolve to Work ID
        work_key: Optional[str] = None

        # Try ISBN first (more reliable), then ISBN13, then title+author search
        if isbn:
            work_key = self._resolve_work_id_by_isbn(isbn)
        if work_key is None and isbn13:
            work_key = self._resolve_work_id_by_isbn(isbn13)
        if work_key is None:
            work_key = self._resolve_work_id_by_search(title, author)

        if work_key is None:
            print(f"  Could not resolve: {title} by {author}")
            return {
                "work_key": None,
                "description": "",
                "subjects": [],
                "subject_places": [],
                "subject_people": [],
                "subject_times": [],
                "cover_url": "",
            }

        # Pass 2: Fetch Work data
        work_data: Optional[dict] = self._fetch_work_data(work_key)
        if work_data is None:
            return {
                "work_key": work_key,
                "description": "",
                "subjects": [],
                "subject_places": [],
                "subject_people": [],
                "subject_times": [],
                "cover_url": "",
            }

        # Extract fields
        description: str = self._extract_description(work_data)
        subjects: list[str] = work_data.get("subjects", [])
        subject_places: list[str] = work_data.get("subject_places", [])
        subject_people: list[str] = work_data.get("subject_people", [])
        subject_times: list[str] = work_data.get("subject_times", [])

        # Build cover URL from first cover ID if available
        cover_url: str = ""
        covers: list[int] = work_data.get("covers", [])
        if covers:
            cover_url = self._build_cover_url(covers[0])

        return {
            "work_key": work_key,
            "description": description,
            "subjects": subjects,
            "subject_places": subject_places,
            "subject_people": subject_people,
            "subject_times": subject_times,
            "cover_url": cover_url,
        }

    def fetch_all(self, df: pd.DataFrame) -> dict[str, dict]:
        """
        Fetch metadata for all books in the DataFrame.
        Uses Book Id as the cache key. Skips books already in cache.

        Args:
            df: DataFrame from DataLoader with columns: Book Id, title, author, isbn, isbn13

        Returns:
            Dict mapping Book Id (str) to metadata dict.
        """
        total: int = len(df)
        fetched: int = 0
        skipped: int = 0

        for _, row in df.iterrows():
            book_id: str = str(row["Book Id"])

            if book_id in self.cache:
                skipped += 1
                continue

            fetched += 1
            print(
                f"[{fetched + skipped}/{total}] Fetching: {row['title']} "
                f"by {row['author']}"
            )

            metadata: dict = self.fetch_metadata_for_book(
                title=row["title"],
                author=row["author"],
                isbn=row["isbn"],
                isbn13=row["isbn13"],
            )
            self.cache[book_id] = metadata

            # Save cache periodically (every 10 books) in case of interruption
            if fetched % 10 == 0:
                self._save_cache()
                print(f"  Cache saved ({fetched} new, {skipped} cached)")

        # Final save
        self._save_cache()
        print(
            f"\nDone. Fetched {fetched} new, {skipped} already cached, "
            f"{total} total."
        )

        return self.cache

    def get_enriched_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge cached metadata into the books DataFrame.

        Adds columns: description, subjects, subject_places,
        subject_people, subject_times, cover_url, work_key.
        """
        metadata_rows: list[dict] = []
        for _, row in df.iterrows():
            book_id: str = str(row["Book Id"])
            meta: dict = self.cache.get(book_id, {})
            metadata_rows.append(
                {
                    "Book Id": row["Book Id"],
                    "description": meta.get("description", ""),
                    "subjects": meta.get("subjects", []),
                    "subject_places": meta.get("subject_places", []),
                    "subject_people": meta.get("subject_people", []),
                    "subject_times": meta.get("subject_times", []),
                    "cover_url": meta.get("cover_url", ""),
                    "work_key": meta.get("work_key", ""),
                }
            )

        meta_df: pd.DataFrame = pd.DataFrame(metadata_rows)
        return df.merge(meta_df, on="Book Id", how="left")


if __name__ == "__main__":
    from data_loader import DataLoader

    loader: DataLoader = DataLoader()
    fetcher: MetadataFetcher = MetadataFetcher()

    print(f"Fetching metadata for {len(loader.df)} books...\n")
    fetcher.fetch_all(loader.df)

    # Show a quick summary
    enriched: pd.DataFrame = fetcher.get_enriched_df(loader.df)
    has_desc: int = enriched["description"].apply(lambda x: len(str(x)) > 0).sum()
    has_subjects: int = enriched["subjects"].apply(lambda x: len(x) > 0).sum()
    print(f"\nBooks with descriptions: {has_desc}/{len(enriched)}")
    print(f"Books with subjects: {has_subjects}/{len(enriched)}")
