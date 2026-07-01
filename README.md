# AI-Powered Spotify Review Analysis

An end-to-end pipeline that collects, cleans, and analyzes user reviews of Spotify from the Google Play Store, Apple App Store, and Reddit — then surfaces patterns in music discovery and recommendation behavior through an interactive Streamlit dashboard.

> **Live Dashboard**: [View on Streamlit Community Cloud](https://shhouraya-spotify-reviews-scraper-dashboardapp-PLACEHOLDER.streamlit.app)
>
> *(Replace the link above with your actual Streamlit URL)*

---

## Screenshots

### Overview Page
<img width="1906" height="921" alt="image" src="https://github.com/user-attachments/assets/7d54785f-20d5-49ce-951f-50b55db78fed" />

### Themes Page
<img width="1906" height="862" alt="image" src="https://github.com/user-attachments/assets/3012ad7b-caaa-48a1-9f5e-e835f7bc54a5" />

### Ask the Data Page
<img width="1916" height="916" alt="image" src="https://github.com/user-attachments/assets/42128fc9-5b7f-4bc3-9b72-ec1264bd83bb" />

### Methodology & Limitations
<img width="1893" height="852" alt="image" src="https://github.com/user-attachments/assets/6f4f1dbf-4b76-46ff-a896-1905230e0c98" />

---

## Research Questions

This project answers six questions about how users experience music discovery and recommendations on Spotify:

1. What struggles do users face when trying to discover new music?
2. What specific recommendation behaviors frustrate users most?
3. How do listening behaviors (playlists, radio, search) shape discovery experiences?
4. Why does repetition occur in recommendations, and how do users respond?
5. Which user segments report the most friction with music discovery?
6. What unmet needs or feature gaps do users describe?

---

## Project Structure

```
spotify-reviews-scraper/
│
├── scrapers/
│   ├── scrape_playstore.py       # Google Play Store scraper
│   ├── scrape_appstore.py        # Apple App Store scraper (iTunes RSS API)
│   ├── scrape_reddit.py          # Reddit scraper (public Atom/RSS feeds)
│   └── validate_stage1.py        # Stage 1 validation tests
│
├── processing/
│   ├── clean.py                  # Deduplication, spam filtering, language detection
│   ├── extract.py                # Groq API thematic extraction (Llama 3.1)
│   ├── aggregate.py              # Embedding-based clustering and aggregation
│   └── validate_stage2.py        # Stage 2 validation tests
│
├── analysis_outputs/
│   ├── theme_summary.json        # Pre-computed themes (dashboard input)
│   └── analyzed_dataset.json     # Per-review extracted fields
│
├── dashboard/
│   └── app.py                    # Streamlit dashboard (4 pages)
│
├── system_architecture.md        # Full pipeline specification
├── requirements.txt
└── .gitignore
```

---

## Pipeline

### Stage 1 — Data Collection
Collects ~400 reviews per source (~1,200 total) using:
- **Google Play Store**: `google-play-scraper` Python library with pagination and retry/backoff
- **Apple App Store**: Direct iTunes RSS API (`itunes.apple.com/.../rss/customerreviews/...`) — no third-party library
- **Reddit**: Public Atom/RSS feeds from r/spotify, r/musicsuggest, r/spotifyplaylist — no OAuth or PRAW required

All sources output a unified schema: `source, id, text, rating, date, extra_metadata`.

### Stage 2 — Cleaning & Deduplication
- Drops entries with fewer than 5 words, blank text, or spam patterns (all-caps, URL-only, repeated single word)
- Exact deduplication via MD5 hash of normalized text
- Near-deduplication via pairwise Jaccard similarity (threshold 0.85)
- Language detection via `langdetect` — non-English reviews are flagged but retained
- **Result**: 932 clean reviews from 1,200 raw

### Stage 3 — Thematic Extraction
Each review is sent to the **Groq API** (`llama-3.1-8b-instant`) with a structured prompt that extracts four fields per review:

| Field | Description |
|---|---|
| `pain_point` | The specific complaint or frustration (string or null) |
| `user_goal` | What the user was trying to accomplish (string or null) |
| `sentiment` | positive / negative / neutral / mixed (always required) |
| `segment_signal` | Who the user appears to be (string or null) |

**Result**: 932/932 records extracted, 0 failures.

### Stage 4 — Clustering & Aggregation
Free-form LLM-extracted labels are grouped into themes using:
- `sentence-transformers` (`all-MiniLM-L6-v2`) to embed unique labels
- KMeans clustering (10 pain point clusters, 8 goal clusters, `random_state=42`)
- Most-frequent-label naming per cluster + manual override dictionary for edge cases
- Segment signals normalized via regex rules into ~25 canonical categories

**Output themes (pain points):** excessive ads, finding new music, app reliability issues, app bugs and broken features, subscription pricing complaints, AI/algorithm dissatisfaction, repetitive recommendations, and more.

### Stage 5 — Dashboard
A 5-page Streamlit app deployed on Streamlit Community Cloud:

| Page | Description |
|---|---|
| **Overview** | Total volume, source breakdown, sentiment distribution, sample reviews |
| **Themes** | Pain points, user goals, and user segments with drill-down detail and example quotes |
| **Question Answers** | Findings mapped to all 6 research questions with supporting data |
| **Ask the Data** | Free-form Q&A — type any question and get an AI-generated answer grounded in the theme data via a live Groq API call |
| **Methodology & Limitations** | Pipeline documentation and caveats |

Pages 1–3 and 5 load exclusively from pre-computed JSON. The "Ask the Data" page makes live Groq API calls at runtime using a key stored in Streamlit secrets.

---

## Setup & Running Locally

### Install dependencies

The `requirements.txt` contains what the dashboard needs (`streamlit`, `pandas`, `groq`). For the full offline pipeline, install the additional packages listed in comments:

```bash
pip install streamlit pandas groq

# For pipeline scripts only:
pip install google-play-scraper langdetect sentence-transformers scikit-learn python-dotenv
```

### Environment variables

Create a `.env` file in the project root (never commit this):

```
GROQ_API_KEY=your_groq_api_key_here
```

The Groq API is free — get a key at [console.groq.com](https://console.groq.com).

**For Streamlit Cloud deployment**: add `GROQ_API_KEY` to the app's secrets via the Streamlit Cloud UI (Manage app → Settings → Secrets). This powers the "Ask the Data" page in the deployed app.

### Run the scrapers (Stage 1)

```bash
python scrapers/scrape_playstore.py
python scrapers/scrape_appstore.py
python scrapers/scrape_reddit.py
```

### Run the pipeline (Stages 2–4)

```bash
python processing/clean.py
python processing/extract.py
python processing/aggregate.py
```

### Launch the dashboard (Stage 5)

```bash
streamlit run dashboard/app.py
```

---

## Ask the Data

The dashboard includes a free-form Q&A interface on the **"Ask the Data"** page. Instead of being limited to the six pre-defined research questions, you can ask anything about the review data:

- *"Which user segment is most frustrated with music discovery?"*
- *"What do users say about Spotify's shuffle algorithm?"*
- *"How do free-tier users differ from premium users in their complaints?"*
- Or any custom question

The page also provides 5 clickable suggested questions to get started quickly.

**How it works**: The question is sent to the Groq API (Llama 3.1 8B) along with the full pre-computed `theme_summary.json` as context — including all theme names, counts, percentages, example quotes, and segment breakdowns. The model is instructed to ground its answer in that data and to say so explicitly if a question falls outside what the reviews cover.

---

## Key Design Decisions

- **Reddit without OAuth**: Reddit's official Data API now requires app approval and triggers a Responsible Builder Policy gate. Instead, this project uses Reddit's public Atom/RSS feeds (`/top.rss`, `/hot.rss`, `/new.rss`), which return 200 without authentication. The `.json` endpoints return HTTP 403 universally.
- **Field-specific denominators**: Theme percentages divide only by the count of reviews where that field is non-null, not total reviews — this avoids artificially deflating percentages.
- **Deterministic clustering**: `sorted(set(labels))` is used before embedding to ensure stable KMeans results across reruns (plain `set()` iteration order is not guaranteed in Python).

---

## Limitations

1. Sample is not population — ~932 reviews from 3 platforms may not represent all Spotify users globally
2. LLM extraction is imperfect — Groq/Llama 3.1 may misclassify sentiment or extract inaccurate pain points
3. Reddit coverage is limited — RSS feeds don't support keyword search, so collection relies on subreddit×sort combinations only
4. No inter-rater reliability — thematic labels were not validated by a second human reviewer
5. Static dashboard — data reflects a single collection snapshot and does not update automatically
