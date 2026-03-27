"""
Unit tests for the DataLoader class.
"""

from pathlib import Path

import pandas as pd
import pytest

from backend.data_loader import DataLoader


def test_loads_correct_number_of_books(data_loader: DataLoader) -> None:
    """DataLoader should load all rows from the CSV."""
    assert len(data_loader.df) == 5


def test_clean_isbn_removes_wrapping(data_loader: DataLoader) -> None:
    """ISBNs wrapped in ="" should be cleaned to plain strings."""
    assert data_loader.df["isbn"].iloc[0] == "0547928246"
    assert data_loader.df["isbn13"].iloc[0] == "9780547928242"


def test_clean_isbn_handles_empty(data_loader: DataLoader) -> None:
    """Empty or missing ISBNs should become empty strings."""
    # Row index 2 (Dune) has empty ISBNs in the sample data
    assert data_loader.df["isbn"].iloc[2] == ""
    assert data_loader.df["isbn13"].iloc[2] == ""


def test_columns_have_correct_types(data_loader: DataLoader) -> None:
    """Cleaned columns should have expected dtypes."""
    assert data_loader.df["my_rating"].dtype == int
    assert data_loader.df["avg_rating"].dtype == float
    assert data_loader.df["read_count"].dtype == int


def test_get_rated_books(data_loader: DataLoader) -> None:
    """get_rated_books should only return books with my_rating > 0."""
    rated: pd.DataFrame = data_loader.get_rated_books()
    assert len(rated) == 3  # The Hobbit (5), 1984 (4), Beloved (3)
    assert all(rated["my_rating"] > 0)


def test_get_to_read_books(data_loader: DataLoader) -> None:
    """get_to_read_books should only return books on the to-read shelf."""
    to_read: pd.DataFrame = data_loader.get_to_read_books()
    assert len(to_read) == 1
    assert to_read.iloc[0]["title"] == "Dune"


def test_get_read_books(data_loader: DataLoader) -> None:
    """get_read_books should only return books on the read shelf."""
    read: pd.DataFrame = data_loader.get_read_books()
    assert len(read) == 3  # The Hobbit, 1984, Beloved
    assert all(read["shelf"] == "read")


def test_get_dnf_books(data_loader: DataLoader) -> None:
    """get_dnf_books should only return did-not-finish books."""
    dnf: pd.DataFrame = data_loader.get_dnf_books()
    assert len(dnf) == 1
    assert dnf.iloc[0]["title"] == "Thinking, Fast and Slow"


def test_get_books_missing_isbn(data_loader: DataLoader) -> None:
    """get_books_missing_isbn should return books with no ISBN-10 or ISBN-13."""
    missing: pd.DataFrame = data_loader.get_books_missing_isbn()
    # Dune and Beloved both have empty ISBNs in sample data
    assert len(missing) == 2
    titles: list[str] = missing["title"].tolist()
    assert "Dune" in titles
    assert "Beloved" in titles


def test_summary_contains_key_info(data_loader: DataLoader) -> None:
    """summary() should include book count and rated count."""
    summary: str = data_loader.summary()
    assert "5 books" in summary
    assert "Rated books: 3" in summary
    assert "To-read books: 1" in summary
    assert "DNF books: 1" in summary
