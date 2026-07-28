# 🎬 Content-Based Movie Recommendation System

A content-based movie recommender built with **Streamlit**, **Pandas**, and **NumPy** — recommendations are generated using a from-scratch bag-of-words vectorizer and cosine similarity, with **no scikit-learn or ML libraries** involved.

🔗 **Live demo:** [content-wisemovie-recommendation-system.streamlit.app](https://content-wisemovie-recommendation-system.streamlit.app/)

## How it works

1. **Load & merge** — `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` are loaded with Pandas and merged on `title`.
2. **Tag building** — for every movie, the app parses the JSON-like `genres`, `keywords`, `cast`, and `crew` columns to pull out genre names, keywords, the top 3 cast members, and the director. These are combined with the split `overview` text into a single `tags` string per movie.
3. **Vectorization** — a vocabulary of up to 5,000 unique words is built from all tags, and each movie's tags are converted into a raw word-count vector against that vocabulary (a hand-rolled bag-of-words, done with plain NumPy arrays).
4. **Cosine similarity** — all vectors are L2-normalized and multiplied (`normalized_vectors @ normalized_vectors.T`) to produce a full movie-by-movie similarity matrix in one vectorized NumPy operation:

   ```
   similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
   ```

5. **Recommend** — picking a movie looks up its row in the similarity matrix, sorts the other movies by score, and returns the top 5 matches.
6. **Visualize** — the top 5 matches and their similarity percentages are plotted as a horizontal bar chart with Plotly.

## Tech stack

| Tool | Role |
|---|---|
| Streamlit | Web UI (dropdown, button, chart rendering) |
| Pandas | Loading, merging, and cleaning the TMDB data |
| NumPy | Vectorization + cosine similarity math |
| Plotly Express | Horizontal bar chart of similarity scores |

## Project structure

```
content-movie-recommendation-system/
├── app.py                    # Streamlit app (data prep, similarity engine, UI)
├── requirements.txt           # streamlit, pandas, numpy, plotly
├── tmdb_5000_movies.csv        # TMDB 5000 movie metadata
├── tmdb_5000_credits.csv       # TMDB 5000 cast & crew data
└── background.avif             # Background image asset
```

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/Supreet37/content-movie-recommendation-system.git
cd content-movie-recommendation-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). The CSV files must stay in the same folder as `app.py` since they're loaded by relative path.

## Usage

1. Pick a movie from the **"Choose a Movie"** dropdown.
2. Click **"Recommend Movies"**.
3. The app displays the top 5 most similar movies as a horizontal bar chart, ranked by cosine similarity percentage.

## Dataset

This project uses the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle, which includes:
- `tmdb_5000_movies.csv` — budget, genres, keywords, overview, popularity, ratings, etc.
- `tmdb_5000_credits.csv` — cast and crew information for each movie.

## Notes & limitations

- Performance is capped by keeping the vocabulary to the first 5,000 unique words found (alphabetically, via `np.unique`) rather than the most frequent ones — this is a simplification worth knowing about if recommendations look off for less common words.
- `@st.cache_data` is used throughout so the (expensive) data loading, tag building, and similarity matrix computation only run once per session rather than on every click.
- No external API calls are made — there are no movie posters, since everything runs on the two local CSVs.

