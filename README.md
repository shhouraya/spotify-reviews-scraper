# AI-Powered Music App Review Analysis

Analyzes user feedback from Play Store, App Store, and Reddit to answer questions about music discovery and recommendation behavior in streaming apps.

## Project Structure

```
/scrapers           - Data collection scripts (Stage 1)
/processing         - Cleaning, deduplication, thematic extraction (Stages 2-3)
/analysis_outputs   - Raw and processed data files
/dashboard          - Streamlit app (Stage 5)
system_architecture.md
requirements.txt
```

## Pipeline

1. **Scrape** reviews from Play Store, App Store, Reddit → `analysis_outputs/raw_*.csv`
2. **Clean & deduplicate** → `analysis_outputs/cleaned_dataset.csv`
3. **Extract themes** via Groq API → `analysis_outputs/analyzed_dataset.json`
4. **Cluster & aggregate** → `analysis_outputs/theme_summary.json`
5. **Dashboard** → deployed Streamlit app

## Setup

```bash
pip install -r requirements.txt
```

Set a `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and `REDDIT_USER_AGENT` environment variable before running the Reddit scraper.

## Running the scrapers

```bash
python scrapers/scrape_playstore.py
python scrapers/scrape_appstore.py
python scrapers/scrape_reddit.py
```
