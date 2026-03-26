# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Book recommendation system. The goal is to implement this project in python as a way to learn more, and create a simple book recommendation system using existing goodreads data.

## Tech stack
| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Data processing** | Python, pandas, scikit-learn | Industry standard for data manipulation and ML |
| **Content engine** | scikit-learn (TfidfVectorizer, cosine_similarity) | Proven, well-documented, lightweight |
| **API / Backend** | TBD (Flask or FastAPI) | Will decide based on needs as we build |
| **Frontend** | Plain HTML / CSS / JavaScript | Simple, no build tools, focus on the recommendation logic |
| **Metadata source** | Open Library API | Free, no auth, good coverage via ISBN lookup |
| **Collaborative filtering** | Surprise library (optional, future) | Purpose-built for rating-based recommender systems |

## Coding Rules
* Always document classes, functions, and potentially confusing variables without being unecessarily verbose.
* Always include type annotations on Python code.
* Write unit tests for all new functionlity.
* Methods should be organized into classes when it makes logical sense to do so.

## Critical Notes
*   If critical information is missing, ask questions first before making assumptions.
*   Do not invent numbers, dates, or policies.