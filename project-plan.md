# Book Recommender — Project Plan

> A living document for the book recommender project. Covers project goals, technical overview of content recommendation, dataset analysis, architecture decisions, and implementation roadmap.

---

## Table of Contents

1. [Project Goals](#project-goals)
2. [How Content Recommendation Works](#how-content-recommendation-works)
   - [The Three Main Approaches](#the-three-main-approaches)
   - [Key Concepts & Vocabulary](#key-concepts--vocabulary)
   - [Combining Multiple Signals](#combining-multiple-signals)
   - [Serendipity & Diversity](#serendipity--diversity)
3. [Dataset Analysis](#dataset-analysis)
4. [Architecture & Design](#architecture--design)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Decisions Log](#decisions-log)
7. [Verification & Testing](#verification--testing)

---

## Project Goals

1. **Learn about content recommendation** — Understand the theory and practice behind recommendation systems, specifically applied to books. Build intuition for how systems like Goodreads, Netflix, and Spotify surface content.

2. **Practice Python skills** — Apply Python to a real data problem: data loading/cleaning, vectorization, similarity computation, and API development.

3. **Build a full-stack application** — Create a working app with a Python backend and a plain HTML/CSS/JS frontend that takes real personal reading data and produces meaningful, personalized book recommendations.

4. **Desired recommendation behavior** — The system should recommend books that:
   - Are related to books the user has rated highly (content similarity)
   - Have strong population ratings (but aren't just bestsellers)
   - Account for behavioral signals like whether a book was finished or abandoned
   - Surface some diversity and surprise, not just "more of the same"

---

## How Content Recommendation Works

### The Three Main Approaches

#### 1. Content-Based Filtering

**Core idea**: "Recommend books similar to what I already like."

This approach examines the *attributes* of items (genre, author, themes, description text) and finds other items with similar attributes. If you rated *The Handmaid's Tale* highly, it looks at that book's features — dystopian fiction, feminist themes, speculative worldbuilding — and finds other books with overlapping features.

**How it works in practice:**

1. **Feature extraction**: Each book's description and metadata are converted into a numerical vector. The most common technique is **TF-IDF** (Term Frequency–Inverse Document Frequency):
   - **Term Frequency (TF)**: How often a word appears in *this* book's description. If "magic" appears 5 times, it has a high TF for that book.
   - **Inverse Document Frequency (IDF)**: How rare the word is across *all* books. "The" appears everywhere (low IDF). "Necromancy" appears rarely (high IDF).
   - **TF-IDF = TF × IDF**: Words that are frequent in this book but rare overall get the highest scores. This automatically filters out common words and highlights distinctive terms.

2. **User profile construction**: The user's taste profile is built as the **weighted average** of feature vectors from books they've rated. A 5-star book contributes more to the profile than a 3-star book. The resulting "user profile vector" lives in the same space as book vectors.

3. **Similarity scoring**: **Cosine similarity** measures how closely an unread book's vector aligns with the user profile vector. Score of 1.0 = identical direction (perfect match), 0.0 = completely unrelated.

**A more modern alternative** to TF-IDF is **embedding-based similarity**. Instead of counting words, you pass descriptions through a language model (like sentence-transformers) which produces a dense vector capturing *meaning*. "A story about a wizard school" and "A tale of young magicians in training" would have very similar embeddings even though they share few exact words. The similarity calculation (cosine similarity) works the same way.

| Strengths | Weaknesses |
|-----------|------------|
| Works with a single user — no other users needed | **Filter bubble** — only recommends things similar to what you've already read |
| Handles new books well (just needs metadata) | Requires good metadata/descriptions for each book |
| Fully personalized and explainable | Can't surface genuinely surprising recommendations |

#### 2. Collaborative Filtering

**Core idea**: "People who liked what you liked also liked..."

This approach ignores book content entirely and finds patterns in how groups of people rate things. It comes in several flavors:

- **User-based**: Build a matrix of users × books filled with ratings. For a target user, find the K most similar users (by rating patterns), then recommend what those similar users enjoyed that the target hasn't read yet.

- **Item-based**: Same matrix, but compute similarity between *books* based on how users co-rate them. If books X and Y are consistently rated similarly by the same people, they're "similar items." To predict your rating for book X, look at books similar to X that you *have* rated. Amazon popularized this approach.

- **Matrix factorization (SVD)**: Decomposes the user-item rating matrix into latent factors. These factors might correspond to hidden dimensions like "literary vs. genre fiction" or "plot-driven vs. character-driven" — the algorithm discovers them automatically. More scalable than user/item-based approaches.

| Strengths | Weaknesses |
|-----------|------------|
| Can surface surprising, unexpected recommendations | **Cold start problem** — can't handle new users or new items with no ratings |
| Doesn't need to know anything about the books themselves | Requires a substantial number of users to find meaningful patterns |
| Captures subtle preference patterns | Less explainable ("users like you also liked X") |

**The Cold Start Problem** — a fundamental challenge in collaborative filtering:
- **New user**: No ratings → can't find similar users. Mitigation: ask them to rate a few books on signup, or fall back to content-based.
- **New item**: Nobody's rated it → it can't appear in recommendations. Mitigation: use content features to bootstrap.
- **New system**: Not enough data for meaningful patterns. Mitigation: start content-based, transition to hybrid as data grows.

#### 3. Hybrid Approaches — What Real Systems Use

Netflix, Spotify, Goodreads, and virtually all production recommendation systems combine both approaches. The most common strategy is **weighted hybridization**: compute a content-based score and a collaborative score independently, then blend them with tunable weights.

Research consistently shows hybrid models outperform either standalone approach.

### Key Concepts & Vocabulary

| Term | Definition |
|------|------------|
| **TF-IDF** | A method to convert text into numbers that highlight distinctive words. Used to represent book descriptions as vectors. |
| **Cosine similarity** | A measure of how similar two vectors are, based on the angle between them. Range: 0 (unrelated) to 1 (identical). |
| **User profile vector** | A weighted average of feature vectors from books the user has rated, representing their tastes in the same space as book features. |
| **Bayesian average** | A technique to compute a "fair" average rating that accounts for the number of ratings, preventing books with very few ratings from dominating. |
| **Cold start** | The challenge of making recommendations when there's insufficient data (new users, new items, or new systems). |
| **Filter bubble** | When a recommender only suggests items similar to past behavior, narrowing the user's exposure over time. |
| **Serendipity** | A recommendation the user wouldn't have found on their own, didn't expect to enjoy, but actually loves. |
| **Latent factors** | Hidden dimensions discovered by matrix factorization that explain rating patterns (e.g., "literary fiction" vs. "genre fiction"). |

### Combining Multiple Signals

Our dataset provides three distinct signals. Here's how each contributes to the final recommendation:

#### Signal 1: Personal Ratings → Content Similarity

Your 87 rated books (1–5 stars) define your taste profile. Books rated 4–5 stars contribute positively and heavily; books rated 1–2 stars contribute as negative signals. The content-based engine computes how similar each unread book is to this profile.

#### Signal 2: Population Average Ratings → Bayesian Average

The "Average Rating" column from Goodreads provides a crowd-sourced quality signal. But raw averages are misleading — a book with 1 rating of 5.0 shouldn't outrank a book with 10,000 ratings averaging 4.5.

The **Bayesian average** (popularized by IMDB's top-250 formula) solves this:

```
weighted_rating = (v / (v + m)) * R + (m / (v + m)) * C
```

Where:
- `R` = the book's average rating
- `v` = number of ratings for this book
- `m` = minimum vote threshold (a tunable parameter, e.g., 50)
- `C` = the global mean rating across all books

When a book has few ratings (`v` is small), its score is pulled toward the global mean `C`. As it accumulates ratings, its own average `R` dominates. This is the first line of defense against naive popularity bias.

**Note**: Our Goodreads export provides Average Rating but not the number of ratings per book. We may need to pull rating counts from the Open Library API, or use Average Rating directly with other dampening techniques.

#### Signal 3: Completion Status → Behavioral Signal

The "Exclusive Shelf" column maps to completion:
- `read` = completed (strong positive signal, especially combined with a high rating)
- `did-not-finish` = abandoned (soft negative signal — even if the book is popular, it didn't work for this user)
- `to-read` = candidate for recommendation
- `currently-reading` = in progress

Completion is arguably a stronger signal than ratings alone. A book you finished and rated 4 stars likely provided more value than one you rated 4 stars but never finished. We use this to:
- Weight completed-and-highly-rated books more heavily in the user profile
- Apply a dampening factor to DNF books in the taste profile
- Use completion rate as an additional feature when scoring

#### The Scoring Formula

```
final_score = (w1 * content_similarity) + (w2 * bayesian_avg_rating) + (w3 * completion_bonus)
```

- `content_similarity` (0–1): How similar the candidate book is to the user's taste profile
- `bayesian_avg_rating` (0–1, normalized): The population quality signal, dampened for fairness
- `completion_bonus`: Adjusts based on patterns — e.g., books by authors the user has finished before get a small boost
- `w1, w2, w3`: Tunable weights. Higher `w1` = more personal, higher `w2` = more popular picks

**Avoiding "just popular books"**: Content similarity acts as a **gate** — a book must pass a minimum content similarity threshold before the popularity score is even considered. This ensures popularity is a tiebreaker among relevant books, not the primary driver.

### Serendipity & Diversity

A recommender that only optimizes for "most likely to enjoy" converges on a narrow slice. Techniques to counter this:

- **Diversification re-ranking**: After generating top-N candidates by score, re-rank to maximize diversity. The **Maximal Marginal Relevance (MMR)** algorithm iteratively selects items that are both relevant to the user *and* dissimilar to items already selected.
- **Exploration-exploitation tradeoff**: Occasionally recommend a book that scores lower on predicted relevance but comes from an underrepresented genre in the user's history.
- **Genre/author quotas**: Ensure no single genre or author dominates more than X% of any recommendation list.
- **Collaborative filtering naturally helps** — it surfaces books popular among like-minded readers even when the content features differ from the user's typical reads.

---

## Dataset Analysis

**Source**: Goodreads CSV export (`data.csv`)
**Size**: 214 books

### Columns Available

| Column | Description | Use in Recommender |
|--------|-------------|-------------------|
| Book Id | Goodreads internal ID | Linking / deduplication |
| Title | Book title | Display |
| Author / Author l-f | Author name | Content feature, display |
| ISBN / ISBN13 | Standard book identifiers | Fetching metadata from Open Library API |
| My Rating | User's 1–5 star rating (0 = unrated) | Core signal for user profile |
| Average Rating | Goodreads community average | Population quality signal |
| Publisher, Binding, Number of Pages | Publication metadata | Potential features |
| Year Published / Original Publication Year | Publication dates | Potential features |
| Date Read / Date Added | Activity timestamps | Recency weighting |
| Exclusive Shelf | read, to-read, did-not-finish, currently-reading | Completion status signal |
| Read Count | Times the user has read the book | Strength of preference |

### Distribution Summary

| Shelf | Count | Role |
|-------|-------|------|
| to-read | 108 | Primary recommendation candidates |
| read | 97 | Training data for user profile |
| did-not-finish | 7 | Negative signal |
| currently-reading | 2 | Exclude from recommendations |

| My Rating | Count | Notes |
|-----------|-------|-------|
| 0 (unrated) | 127 | Mostly to-read shelf |
| 1 star | 3 | Strong negative |
| 2 stars | 4 | Negative |
| 3 stars | 20 | Neutral/mild positive |
| 4 stars | 41 | Positive |
| 5 stars | 19 | Strong positive |

### Data Gap: No Descriptions or Genres

The Goodreads export does not include book descriptions, genres, or subject tags. These are essential for content-based filtering. We will enrich the dataset by querying the **Open Library API** (free, no authentication required) using ISBNs to fetch:
- Book descriptions/summaries
- Subject tags and genres
- Additional metadata (cover images, etc.)

### Which Books Do We Fetch Metadata For?

We can't pull metadata for "all books" — there are millions. Instead, we use a tiered approach:

**Tier 1 — Your existing Goodreads library (v1, ~214 API calls)**
Fetch metadata for every book already in the Goodreads export. This is essential because:
- The 87 rated books need descriptions to build the user taste profile
- The 108 to-read books need descriptions so we can score them against that profile
- This is a small, finite set — very manageable

With Tier 1, the recommender answers: **"Which of my to-read books should I read next?"** — already a useful and complete v1.

**Tier 2 — Discovering new books (future enhancement)**
Once the taste profile is built from Tier 1, we can expand the candidate pool:
- **Author/subject adjacency**: Query Open Library's search API for books in the user's top genres or by authors similar to their favorites — a curated expansion, not "all books"
- **Curated seed lists**: Pull from award winner lists, popular Goodreads lists, or genre-specific collections as a bounded discovery pool
- This turns the recommender into: **"What books should I add to my to-read list?"**

**Decision**: Start with Tier 1 only. Expand to Tier 2 as a later enhancement once the core engine is working.

### How We Cache Enriched Data

For this project's scale (~200–1000 books), a **JSON file** is the right choice — simpler than a database and perfectly adequate:

- After fetching from Open Library, save all enriched data to `data/enriched_books.json`
- On startup, the app checks if this file exists and loads from it instead of re-fetching
- To refresh or add books, a simple script re-fetches and overwrites the cache
- The JSON file is human-readable and easy to inspect/debug during development

A database (e.g., SQLite) would be warranted if we had tens of thousands of books, multiple users, or needed frequent individual record updates. We can always migrate later if the project grows.

---

## Open Library API Reference

### Key Concept: Works vs Editions

Open Library has a hierarchy: **Works** (the abstract book, e.g., "1984") → **Editions** (specific published versions, each with its own ISBN). A single Work can have hundreds of Editions.

**Critical finding**: Descriptions and subjects live primarily on the **Work**, not the Edition. Many editions have no description at all. Our fetcher must always resolve to the Work level for descriptions and subjects.

### Rate Limits

| Scenario | Rate Limit |
|----------|-----------|
| Unidentified requests (no User-Agent) | 1 request/second |
| Identified requests (with User-Agent + email) | **3 requests/second** |

We must include: `User-Agent: bookrecommend/1.0 maia.muzumdar@gmail.com`

For 214 books needing ~428 calls (2 per book), budget ~2.5 minutes total.

### APIs We Use

#### 1. ISBN Lookup — Resolve books with ISBNs to Work IDs

**Endpoint**: `GET https://openlibrary.org/isbn/{ISBN}.json`

Returns an **Edition** record. The key field we need is `works[0].key` which gives the Work ID.

**Key fields returned**: `title`, `publishers`, `publish_date`, `number_of_pages`, `isbn_10`, `isbn_13`, `covers` (array of cover IDs), `works` (array with Work key), `description` (sometimes, but unreliable at edition level).

#### 2. Search API — Find books by title+author (when ISBN is missing)

**Endpoint**: `GET https://openlibrary.org/search.json`

| Parameter | Description |
|-----------|-------------|
| `title` | Search by title |
| `author` | Search by author name |
| `fields` | Comma-separated fields to return (reduces response size) |
| `limit` | Number of results (default 10) |

**Optimal field list**: `fields=key,title,author_name,subject,cover_i,first_publish_year`

**Example**: `GET /search.json?title=Beloved&author=Toni+Morrison&fields=key,title,author_name,subject,cover_i&limit=1`

Returns `docs[0].key` as the Work ID (e.g., `/works/OL50548W`). The `subject` field is also returned inline from Search, so we get subjects without a second call — though the Works API gives richer subject data.

#### 3. Works API — Primary source for descriptions & subjects

**Endpoint**: `GET https://openlibrary.org/works/{WORK_ID}.json`

| Field | Type | Notes |
|-------|------|-------|
| `description` | string OR `{type, value}` object | **Polymorphic** — must handle both formats |
| `subjects` | array of strings | Genre/topic tags (e.g., `["Fiction", "Dystopia"]`) |
| `subject_places` | array of strings | Geographic settings |
| `subject_people` | array of strings | Named characters/people |
| `subject_times` | array of strings | Time periods |
| `covers` | array of integers | Cover image IDs |
| `excerpts` | array | Text excerpts |

**Description format quirk**: The `description` field can be:
- A plain string: `"description": "This is the description..."`
- An object: `"description": {"type": "/type/text", "value": "This is the description..."}`

Must handle both in code.

#### 4. Covers API — No API call needed, just URL construction

```
https://covers.openlibrary.org/b/id/{COVER_ID}-{SIZE}.jpg
https://covers.openlibrary.org/b/isbn/{ISBN}-{SIZE}.jpg
```

Sizes: `S` (small), `M` (medium), `L` (large). Cover IDs come from the `covers` field in Works or `cover_i` from Search.

### Fetching Strategy (Two-Pass)

**Pass 1 — Resolve all books to Work IDs:**
- Books WITH ISBN: `GET /isbn/{isbn}.json` → extract `works[0].key`
- Books WITHOUT ISBN: `GET /search.json?title=X&author=Y&limit=1` → extract `docs[0].key`

**Pass 2 — Fetch full Work data:**
- For ALL books: `GET /works/{work_id}.json` → extract `description`, `subjects`, `covers`

### Data Quality Notes

- **Subjects are noisy**: Include reading levels, collection IDs, NYT bestseller tags, Library of Congress call numbers, foreign-language translations. Filtering will be needed.
- **Descriptions vary wildly**: From single sentences to multi-paragraph summaries. Some include attribution like "--back cover".
- **Some books will have no description** on Open Library. Fallback: use `first_sentence` from Search, or leave blank.
- **Author names** in search results are arrays (books can have multiple authors).

---

## Architecture & Design

### Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Data processing** | Python, pandas, scikit-learn | Industry standard for data manipulation and ML |
| **Content engine** | scikit-learn (TfidfVectorizer, cosine_similarity) | Proven, well-documented, lightweight |
| **API / Backend** | TBD (Flask or FastAPI) | Will decide based on needs as we build |
| **Frontend** | Plain HTML / CSS / JavaScript | Simple, no build tools, focus on the recommendation logic |
| **Metadata source** | Open Library API | Free, no auth, good coverage via ISBN lookup |
| **Collaborative filtering** | Surprise library (optional, future) | Purpose-built for rating-based recommender systems |
| **Testing** | pytest, requests-mock, pytest-cov | Modern Python testing standard; see Testing section below |

### Project Structure (Planned)

```
book-recommend/
├── project-plan.md          # This document
├── pytest.ini               # Pytest configuration
├── backend/
│   ├── __init__.py          # Package marker for imports
│   ├── app.py               # API server
│   ├── recommender.py       # Core recommendation engine
│   ├── data_loader.py       # Load and clean Goodreads data
│   ├── metadata_fetcher.py  # Open Library API integration
│   └── requirements.txt     # Python dependencies
├── tests/
│   ├── __init__.py          # Package marker for imports
│   ├── conftest.py          # Shared test fixtures
│   ├── test_data_loader.py  # DataLoader unit tests
│   └── test_metadata_fetcher.py  # MetadataFetcher unit tests
├── frontend/
│   ├── index.html           # Main page
│   ├── style.css            # Styling
│   └── app.js               # Frontend logic, API calls
└── data/
    ├── data.csv              # Goodreads export
    └── enriched_books.json   # Cached enriched book metadata (gitignored)
```

---

## Implementation Roadmap

### Phase 1: Data Loading & Enrichment
- Load Goodreads CSV with pandas
- Clean and normalize the data (handle unrated books, parse dates, etc.)
- Build metadata fetcher to query Open Library API by ISBN
- Cache enriched data locally (descriptions, genres, subjects) to avoid repeated API calls
- Explore and validate the enriched dataset

### Phase 2: Content-Based Recommendation Engine
- Build TF-IDF vectors from book descriptions using scikit-learn's `TfidfVectorizer`
- Construct the user taste profile as a weighted average of rated books' vectors
- Compute cosine similarity between the user profile and all unread books
- Implement Bayesian average for population ratings
- Combine into the hybrid scoring formula
- Test with the to-read shelf: do the top recommendations make intuitive sense?

### Phase 3: Backend API
- Set up Python backend (Flask or FastAPI)
- Create endpoints:
  - `GET /recommendations` — returns ranked book recommendations with scores
  - `GET /books` — returns all books with metadata and ratings
  - `POST /rate` — allows updating a book's rating (for future interactivity)
- Serve recommendation results as JSON

### Phase 4: Frontend
- Build a clean, simple HTML/CSS/JS interface
- Display recommended books with title, author, cover, scores, and explanation
- Add weight adjustment controls (sliders for content vs. popularity balance)
- Show the user's reading history and ratings

### Phase 5: Refinement & Polish
- Add diversity re-ranking to avoid homogeneous recommendation lists
- Tune scoring weights based on qualitative review of results
- Handle edge cases (missing metadata, books with no description, etc.)
- Optional: add collaborative filtering via the Surprise library if we want to expand

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-26 | Frontend: plain HTML/CSS/JS | Simpler, no build tools, keeps focus on the recommendation logic |
| 2026-03-26 | Backend: TBD | Will decide between Flask and FastAPI as we build |
| 2026-03-26 | Metadata source: Open Library API | Free, no auth, supports ISBN lookup |
| 2026-03-26 | Start with content-based + popularity hybrid | Collaborative filtering needs more users; content-based works well for single-user scenario |
| 2026-03-26 | v1 scope: rank the to-read shelf only (Tier 1) | Fetch metadata for existing 214 books; expanding to discover new books is a Tier 2 enhancement |
| 2026-03-26 | Cache enriched data as JSON file (`data/enriched_books.json`) | Simple, human-readable, adequate for ~200–1000 books; can migrate to SQLite if project grows |
| 2026-03-26 | Testing: pytest + requests-mock + pytest-cov | Modern standard, clean fixture system, native HTTP mocking for requests library |
| 2026-03-26 | Test data: in-memory fixtures, not external files | Fast, self-contained, no file pollution; temp dirs for file-based tests |

---

## Testing

### Framework Choices & Rationale

| Tool | Role | Why we chose it |
|------|------|----------------|
| **pytest** | Test framework | Modern Python standard (~52% adoption). Uses plain `assert` instead of unittest's verbose `self.assertEqual`. Powerful fixture system with dependency injection and scoping. Automatic test discovery. Rich plugin ecosystem. |
| **requests-mock** | HTTP mocking | Integrates natively with pytest as a fixture parameter — no decorators or context managers needed. Cleaner than `unittest.mock` for mocking `requests.Session` calls. Purpose-built for the `requests` library we use. |
| **pytest-cov** | Coverage reporting | Tracks which lines of backend code are exercised by tests. Integrates directly with pytest via `--cov` flag. |
| **pandas.testing** | DataFrame assertions | Built-in pandas module with `assert_frame_equal` and `assert_series_equal` for comparing DataFrames in tests. |

**Why not unittest?** unittest requires subclassing `TestCase`, using special assertion methods (`assertEqual`, `assertTrue`), and verbose `setUp`/`tearDown`. pytest achieves the same with less boilerplate and better error messages.

**Why not VCR.py for HTTP mocking?** VCR.py records real API responses to "cassette" files and replays them. Good for integration tests, but adds file management overhead. For unit tests, requests-mock is simpler — we define expected responses inline and don't need to manage cassette files.

### How to Run Tests

```bash
# From the book-recommend/ directory:
pytest -v                    # Run all tests with verbose output
pytest -v --cov=backend      # Run with coverage report
pytest tests/test_data_loader.py  # Run tests for one module
```

### Test Data Strategy: In-Memory Fixtures

Tests use **in-memory DataFrames** created in `tests/conftest.py` rather than external CSV files. A small sample DataFrame (3-5 rows) covers the key scenarios:
- A rated, completed book with a valid ISBN
- A rated, completed book with no ISBN
- An unrated to-read book
- A did-not-finish book
- Edge cases: empty ISBNs, wrapped ISBN format (`="123"`)

For tests that need files on disk (e.g., testing DataLoader's CSV loading or MetadataFetcher's cache persistence), fixtures write the sample data to a temporary directory that's automatically cleaned up after each test.

For HTTP tests, **requests-mock** intercepts outgoing requests and returns predefined responses — no real API calls are made. This makes tests fast, deterministic, and offline-safe.

### What We Test

#### DataLoader (`tests/test_data_loader.py`)

| Test | What it verifies |
|------|-----------------|
| `test_loads_correct_number_of_books` | CSV parsing produces expected row count |
| `test_clean_isbn_removes_wrapping` | `="0547928246"` → `"0547928246"` |
| `test_clean_isbn_handles_empty` | Empty/NaN ISBN → `""` |
| `test_columns_have_correct_types` | `my_rating` is int, `avg_rating` is float |
| `test_get_rated_books` | Only returns books with `my_rating > 0` |
| `test_get_to_read_books` | Only returns `shelf == "to-read"` |
| `test_get_read_books` | Only returns `shelf == "read"` |
| `test_get_dnf_books` | Only returns `shelf == "did-not-finish"` |
| `test_get_books_missing_isbn` | Returns books with both `isbn` and `isbn13` empty |
| `test_summary_contains_key_info` | Summary string includes book count, rated count |

#### MetadataFetcher (`tests/test_metadata_fetcher.py`)

| Test | What it verifies |
|------|-----------------|
| `test_extract_description_string` | Handles plain string description |
| `test_extract_description_dict` | Handles `{"type": ..., "value": ...}` format |
| `test_extract_description_missing` | No description field → `""` |
| `test_clean_title_strips_series` | `"Book (Series, #2)"` → `"Book"` |
| `test_clean_title_strips_subtitle` | `"Book: Long Subtitle"` → `"Book"` |
| `test_clean_title_no_change` | Simple title stays unchanged |
| `test_build_cover_url` | Generates correct cover URL from ID |
| `test_resolve_work_id_by_isbn` | Mocked ISBN endpoint → correct work key |
| `test_resolve_work_id_by_isbn_not_found` | 404 → `None` |
| `test_resolve_work_id_by_search` | Mocked search → correct work key |
| `test_request_handles_rate_limit` | 429 then 200 → retries and succeeds |
| `test_request_handles_network_error` | Connection error → `None` |
| `test_cache_saves_and_loads` | Save cache, new instance loads it correctly |
| `test_fetch_metadata_for_book_full_flow` | Full two-pass flow with mocked endpoints |

### Fixture Architecture

Fixtures are defined in `tests/conftest.py` and automatically injected into test functions by pytest:

```
sample_goodreads_df        → In-memory DataFrame (3-5 rows)
    ↓
temp_csv_file              → Writes DataFrame to temp CSV file
    ↓
data_loader                → DataLoader initialized with temp CSV

sample_metadata_cache      → In-memory dict mimicking enriched_books.json
    ↓
temp_cache_file            → Writes dict to temp JSON file
    ↓
metadata_fetcher           → MetadataFetcher initialized with temp cache
```

Fixtures use `tempfile.TemporaryDirectory` for file-based tests, ensuring cleanup after each test. The `requests_mock` fixture is provided automatically by the requests-mock library.

---

## Verification — Recommender Quality

### How to validate the recommender is working well

1. **Sanity check**: Do the top-10 recommendations include books that intuitively match the user's reading history? (e.g., if the user loves literary fiction, are the top picks literary fiction?)
2. **Diversity check**: Are recommendations spread across multiple genres/authors, or clustered around one?
3. **Popularity check**: Are the recommendations genuinely good books (high average rating) without being just the most popular books on Goodreads?
4. **Negative signal check**: Are DNF books and their close relatives ranked lower?
5. **Weight tuning**: Adjusting w1/w2/w3 should visibly shift the character of recommendations (more w1 = more niche/personal, more w2 = more popular/mainstream)
6. **End-to-end**: Load the app in a browser, see recommendations, adjust sliders, verify the list updates accordingly
