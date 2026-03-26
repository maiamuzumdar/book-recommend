"""
Load and clean Goodreads CSV export data.
"""

import os

import pandas as pd

# Default path to the Goodreads CSV export
DATA_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
DEFAULT_CSV_PATH: str = os.path.join(DATA_DIR, "data.csv")


class DataLoader:
    """Loads, cleans, and provides access to Goodreads export data."""

    def __init__(self, csv_path: str = DEFAULT_CSV_PATH) -> None:
        self.csv_path: str = csv_path
        self.df: pd.DataFrame = self._load_and_clean()

    def _clean_isbn(self, value: str) -> str:
        """Remove Goodreads ="" wrapping from ISBN fields."""
        if pd.isna(value):
            return ""
        return str(value).replace('="', "").replace('"', "").strip()

    def _load_and_clean(self) -> pd.DataFrame:
        """
        Load the Goodreads CSV and return a cleaned DataFrame.

        Columns added/cleaned:
          - isbn: cleaned ISBN-10
          - isbn13: cleaned ISBN-13
          - shelf: one of 'read', 'to-read', 'did-not-finish', 'currently-reading'
          - my_rating: integer 0-5 (0 = unrated)
          - avg_rating: float, community average rating
        """
        df: pd.DataFrame = pd.read_csv(self.csv_path)

        # Clean ISBNs (Goodreads wraps them in ="...")
        df["isbn"] = df["ISBN"].apply(self._clean_isbn)
        df["isbn13"] = df["ISBN13"].apply(self._clean_isbn)

        # Normalize column names we'll use frequently
        df["my_rating"] = df["My Rating"].astype(int)
        df["avg_rating"] = df["Average Rating"].astype(float)
        df["shelf"] = df["Exclusive Shelf"].str.strip()
        df["title"] = df["Title"].str.strip()
        df["author"] = df["Author"].str.strip()
        df["num_pages"] = pd.to_numeric(df["Number of Pages"], errors="coerce")
        df["year_published"] = pd.to_numeric(
            df["Original Publication Year"], errors="coerce"
        )
        df["read_count"] = df["Read Count"].astype(int)

        return df

    def get_all_books(self) -> pd.DataFrame:
        """Return all books."""
        return self.df

    def get_rated_books(self) -> pd.DataFrame:
        """Return books the user has rated (1-5 stars)."""
        return self.df[self.df["my_rating"] > 0]

    def get_to_read_books(self) -> pd.DataFrame:
        """Return books on the to-read shelf."""
        return self.df[self.df["shelf"] == "to-read"]

    def get_read_books(self) -> pd.DataFrame:
        """Return books on the read shelf."""
        return self.df[self.df["shelf"] == "read"]

    def get_dnf_books(self) -> pd.DataFrame:
        """Return books the user did not finish."""
        return self.df[self.df["shelf"] == "did-not-finish"]

    def get_books_missing_isbn(self) -> pd.DataFrame:
        """Return books with no ISBN-10 or ISBN-13."""
        return self.df[(self.df["isbn"] == "") & (self.df["isbn13"] == "")]

    def summary(self) -> str:
        """Return a human-readable summary of the loaded data."""
        lines: list[str] = [
            f"Loaded {len(self.df)} books from {self.csv_path}",
            "",
            "Shelf distribution:",
            self.df["shelf"].value_counts().to_string(),
            "",
            "Rating distribution:",
            self.df["my_rating"].value_counts().sort_index().to_string(),
            "",
            f"Rated books: {len(self.get_rated_books())}",
            f"To-read books: {len(self.get_to_read_books())}",
            f"DNF books: {len(self.get_dnf_books())}",
            f"Books with no ISBN: {len(self.get_books_missing_isbn())}",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    loader: DataLoader = DataLoader()
    print(loader.summary())
