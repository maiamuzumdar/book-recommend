"""
Unit tests for the MetadataFetcher class.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from requests.exceptions import ConnectionError
from requests_mock import Mocker

from backend.metadata_fetcher import MetadataFetcher


# --- Description extraction ---


def test_extract_description_string(metadata_fetcher: MetadataFetcher) -> None:
    """Should extract a plain string description."""
    result: str = metadata_fetcher._extract_description(
        {"description": "A great book about adventure."}
    )
    assert result == "A great book about adventure."


def test_extract_description_dict(metadata_fetcher: MetadataFetcher) -> None:
    """Should extract description from the polymorphic dict format."""
    result: str = metadata_fetcher._extract_description(
        {"description": {"type": "/type/text", "value": "A dystopian novel."}}
    )
    assert result == "A dystopian novel."


def test_extract_description_missing(metadata_fetcher: MetadataFetcher) -> None:
    """Should return empty string when no description field exists."""
    result: str = metadata_fetcher._extract_description({})
    assert result == ""


# --- Title cleaning ---


def test_clean_title_strips_series(metadata_fetcher: MetadataFetcher) -> None:
    """Should strip parenthetical series info from titles."""
    result: str = metadata_fetcher._clean_title_for_search(
        "The Testaments (The Handmaid's Tale, #2)"
    )
    assert result == "The Testaments"


def test_clean_title_strips_subtitle(metadata_fetcher: MetadataFetcher) -> None:
    """Should strip subtitles after a colon."""
    result: str = metadata_fetcher._clean_title_for_search(
        "Atomic Habits: An Easy & Proven Way to Build Good Habits"
    )
    assert result == "Atomic Habits"


def test_clean_title_no_change(metadata_fetcher: MetadataFetcher) -> None:
    """Should leave simple titles unchanged."""
    result: str = metadata_fetcher._clean_title_for_search("Beloved")
    assert result == "Beloved"


# --- Cover URL ---


def test_build_cover_url(metadata_fetcher: MetadataFetcher) -> None:
    """Should generate correct Open Library cover URL."""
    url: str = metadata_fetcher._build_cover_url(12345, "M")
    assert url == "https://covers.openlibrary.org/b/id/12345-M.jpg"


def test_build_cover_url_default_size(metadata_fetcher: MetadataFetcher) -> None:
    """Should default to medium size."""
    url: str = metadata_fetcher._build_cover_url(99999)
    assert url == "https://covers.openlibrary.org/b/id/99999-M.jpg"


# --- API request handling ---


@patch("backend.metadata_fetcher.REQUEST_DELAY", 0)
def test_resolve_work_id_by_isbn(
    metadata_fetcher: MetadataFetcher, requests_mock: Mocker
) -> None:
    """Should resolve an ISBN to a Work ID via the ISBN endpoint."""
    requests_mock.get(
        "https://openlibrary.org/isbn/0547928246.json",
        json={"works": [{"key": "/works/OL45883W"}]},
    )

    result: str = metadata_fetcher._resolve_work_id_by_isbn("0547928246")
    assert result == "/works/OL45883W"


@patch("backend.metadata_fetcher.REQUEST_DELAY", 0)
def test_resolve_work_id_by_isbn_not_found(
    metadata_fetcher: MetadataFetcher, requests_mock: Mocker
) -> None:
    """Should return None when ISBN endpoint returns 404."""
    requests_mock.get(
        "https://openlibrary.org/isbn/0000000000.json",
        status_code=404,
    )

    result = metadata_fetcher._resolve_work_id_by_isbn("0000000000")
    assert result is None


@patch("backend.metadata_fetcher.REQUEST_DELAY", 0)
def test_resolve_work_id_by_search(
    metadata_fetcher: MetadataFetcher, requests_mock: Mocker
) -> None:
    """Should resolve a title+author to a Work ID via the Search API."""
    requests_mock.get(
        "https://openlibrary.org/search.json",
        json={
            "numFound": 1,
            "docs": [{"key": "/works/OL50548W", "title": "Beloved"}],
        },
    )

    result: str = metadata_fetcher._resolve_work_id_by_search(
        "Beloved", "Toni Morrison"
    )
    assert result == "/works/OL50548W"


@patch("backend.metadata_fetcher.REQUEST_DELAY", 0)
def test_request_handles_rate_limit(
    metadata_fetcher: MetadataFetcher, requests_mock: Mocker
) -> None:
    """Should retry once after a 429 rate limit response."""
    requests_mock.get(
        "https://openlibrary.org/test.json",
        [
            {"status_code": 429},
            {"status_code": 200, "json": {"success": True}},
        ],
    )

    with patch("backend.metadata_fetcher.time.sleep"):
        result = metadata_fetcher._request("https://openlibrary.org/test.json")
    assert result == {"success": True}


@patch("backend.metadata_fetcher.REQUEST_DELAY", 0)
def test_request_handles_network_error(
    metadata_fetcher: MetadataFetcher, requests_mock: Mocker
) -> None:
    """Should return None on a network error."""
    requests_mock.get(
        "https://openlibrary.org/test.json",
        exc=ConnectionError("Connection refused"),
    )

    result = metadata_fetcher._request("https://openlibrary.org/test.json")
    assert result is None


# --- Caching ---


@patch("backend.metadata_fetcher.REQUEST_DELAY", 0)
def test_cache_saves_and_loads(temp_data_dir: Path) -> None:
    """Cache should persist to disk and load on new instance creation."""
    cache_path: str = str(temp_data_dir / "test_cache.json")
    fetcher: MetadataFetcher = MetadataFetcher(cache_path=cache_path)

    # Manually populate and save cache
    fetcher.cache["42"] = {
        "work_key": "/works/OL999W",
        "description": "Test book",
        "subjects": ["Testing"],
    }
    fetcher._save_cache()

    # Create a new instance — should load the saved cache
    fetcher2: MetadataFetcher = MetadataFetcher(cache_path=cache_path)
    assert "42" in fetcher2.cache
    assert fetcher2.cache["42"]["work_key"] == "/works/OL999W"
    assert fetcher2.cache["42"]["description"] == "Test book"


# --- Full flow ---


@patch("backend.metadata_fetcher.REQUEST_DELAY", 0)
def test_fetch_metadata_for_book_full_flow(
    metadata_fetcher: MetadataFetcher, requests_mock: Mocker
) -> None:
    """Should resolve ISBN → Work ID → full metadata in two passes."""
    # Pass 1: ISBN → Edition with work key
    requests_mock.get(
        "https://openlibrary.org/isbn/0547928246.json",
        json={"works": [{"key": "/works/OL45883W"}]},
    )

    # Pass 2: Work ID → full metadata
    requests_mock.get(
        "https://openlibrary.org/works/OL45883W.json",
        json={
            "description": "A hobbit goes on an adventure.",
            "subjects": ["Fantasy", "Adventure"],
            "subject_places": ["Middle-earth"],
            "subject_people": ["Bilbo Baggins"],
            "subject_times": [],
            "covers": [12345],
        },
    )

    result: dict = metadata_fetcher.fetch_metadata_for_book(
        title="The Hobbit",
        author="J.R.R. Tolkien",
        isbn="0547928246",
        isbn13="9780547928242",
    )

    assert result["work_key"] == "/works/OL45883W"
    assert result["description"] == "A hobbit goes on an adventure."
    assert "Fantasy" in result["subjects"]
    assert result["subject_places"] == ["Middle-earth"]
    assert result["cover_url"] == "https://covers.openlibrary.org/b/id/12345-M.jpg"
