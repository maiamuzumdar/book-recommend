"""
Shared test fixtures for the book recommender test suite.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from backend.data_loader import DataLoader
from backend.metadata_fetcher import MetadataFetcher


@pytest.fixture
def sample_goodreads_df() -> pd.DataFrame:
    """
    Small in-memory DataFrame mimicking a Goodreads CSV export.
    Covers key scenarios: rated/unrated books, different shelves,
    wrapped/empty ISBNs.
    """
    return pd.DataFrame(
        {
            "Book Id": [1, 2, 3, 4, 5],
            "Title": [
                "The Hobbit",
                "1984",
                "Dune",
                "Thinking, Fast and Slow",
                "Beloved",
            ],
            "Author": [
                "J.R.R. Tolkien",
                "George Orwell",
                "Frank Herbert",
                "Daniel Kahneman",
                "Toni Morrison",
            ],
            "Author l-f": [
                "Tolkien, J.R.R.",
                "Orwell, George",
                "Herbert, Frank",
                "Kahneman, Daniel",
                "Morrison, Toni",
            ],
            "Additional Authors": ["", "", "", "", ""],
            "ISBN": [
                '="0547928246"',
                '="0451524935"',
                '=""',
                '="0374275637"',
                '=""',
            ],
            "ISBN13": [
                '="9780547928242"',
                '="9780451524935"',
                '=""',
                '="9780374275631"',
                '=""',
            ],
            "My Rating": [5, 4, 0, 0, 3],
            "Average Rating": [4.28, 4.19, 4.25, 4.17, 4.38],
            "Publisher": [
                "Houghton Mifflin",
                "Signet Classic",
                "Ace",
                "FSG",
                "Vintage",
            ],
            "Binding": [
                "Paperback",
                "Paperback",
                "Paperback",
                "Hardcover",
                "Paperback",
            ],
            "Number of Pages": [310, 328, 688, 499, 324],
            "Year Published": [2012, 1961, 1990, 2011, 2004],
            "Original Publication Year": [1937, 1949, 1965, 2011, 1987],
            "Date Read": ["2024/01/15", "2024/03/20", "", "", "2023/12/01"],
            "Date Added": [
                "2024/01/01",
                "2024/03/01",
                "2024/04/01",
                "2024/02/15",
                "2023/11/15",
            ],
            "Bookshelves": ["", "", "to-read", "did-not-finish", ""],
            "Bookshelves with positions": [
                "",
                "",
                "to-read (#1)",
                "did-not-finish (#1)",
                "",
            ],
            "Exclusive Shelf": [
                "read",
                "read",
                "to-read",
                "did-not-finish",
                "read",
            ],
            "My Review": ["", "", "", "", ""],
            "Spoiler": ["", "", "", "", ""],
            "Private Notes": ["", "", "", "", ""],
            "Read Count": [2, 1, 0, 0, 1],
            "Owned Copies": [1, 0, 0, 0, 1],
        }
    )


@pytest.fixture
def sample_metadata_cache() -> dict[str, dict]:
    """Sample cached metadata mimicking enriched_books.json."""
    return {
        "1": {
            "work_key": "/works/OL45883W",
            "description": "A hobbit goes on an unexpected adventure.",
            "subjects": ["Fantasy", "Adventure", "Fiction"],
            "subject_places": ["Middle-earth"],
            "subject_people": ["Bilbo Baggins"],
            "subject_times": [],
            "cover_url": "https://covers.openlibrary.org/b/id/12345-M.jpg",
        },
        "2": {
            "work_key": "/works/OL1168083W",
            "description": "A dystopian novel about totalitarian government.",
            "subjects": ["Dystopian", "Politics", "Fiction"],
            "subject_places": ["Oceania"],
            "subject_people": ["Winston Smith"],
            "subject_times": ["1984"],
            "cover_url": "https://covers.openlibrary.org/b/id/67890-M.jpg",
        },
    }


@pytest.fixture
def temp_data_dir() -> Path:
    """Temporary directory for file-based tests, cleaned up automatically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_csv_file(
    temp_data_dir: Path, sample_goodreads_df: pd.DataFrame
) -> Path:
    """Write sample Goodreads data to a temp CSV file."""
    csv_path: Path = temp_data_dir / "test_books.csv"
    sample_goodreads_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_cache_file(
    temp_data_dir: Path, sample_metadata_cache: dict[str, dict]
) -> Path:
    """Write sample metadata cache to a temp JSON file."""
    cache_path: Path = temp_data_dir / "test_cache.json"
    with open(cache_path, "w") as f:
        json.dump(sample_metadata_cache, f)
    return cache_path


@pytest.fixture
def data_loader(temp_csv_file: Path) -> DataLoader:
    """DataLoader initialized with sample test data."""
    return DataLoader(csv_path=str(temp_csv_file))


@pytest.fixture
def metadata_fetcher(temp_data_dir: Path) -> MetadataFetcher:
    """MetadataFetcher with an empty cache in a temp directory."""
    cache_path: Path = temp_data_dir / "test_cache.json"
    return MetadataFetcher(cache_path=str(cache_path))


@pytest.fixture
def metadata_fetcher_with_cache(temp_cache_file: Path) -> MetadataFetcher:
    """MetadataFetcher pre-loaded with sample cached metadata."""
    return MetadataFetcher(cache_path=str(temp_cache_file))
