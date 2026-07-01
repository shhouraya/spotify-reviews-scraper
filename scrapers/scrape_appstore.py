"""
App Store scraper for Spotify reviews using the iTunes RSS API directly.
Outputs: analysis_outputs/raw_appstore.csv
Columns: source, id, text, rating, date, extra_metadata
"""

import json
import time
import urllib.request
import urllib.error
import pandas as pd
from pathlib import Path

APP_ID = "324684580"
COUNTRY = "us"
TARGET_COUNT = 400
MAX_PAGES = 10  # iTunes RSS caps at page 10, 50 reviews each = 500 max
OUTPUT_PATH = Path(__file__).parent.parent / "analysis_outputs" / "raw_appstore.csv"


def fetch_page(page: int, retries: int = 5) -> list[dict]:
    url = (
        f"https://itunes.apple.com/{COUNTRY}/rss/customerreviews/"
        f"page={page}/id={APP_ID}/sortby=mostrecent/json"
    )
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "spotify-review-scraper/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            entries = data.get("feed", {}).get("entry", [])
            # First entry on page 1 is app metadata, not a review
            if page == 1 and entries and "im:rating" not in entries[0]:
                entries = entries[1:]
            return entries
        except urllib.error.HTTPError as e:
            if e.code == 400:
                return []  # No more pages
            wait = 2 ** (attempt + 1)
            print(f"  HTTP {e.code} on page {page}, retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"  Error on page {page} ({e}), retrying in {wait}s...")
            time.sleep(wait)
    return []


def scrape(count: int = TARGET_COUNT) -> pd.DataFrame:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scraping up to {count} App Store reviews for app ID {APP_ID}...")

    rows = []
    seen_ids = set()
    pages_needed = min(MAX_PAGES, -(-count // 50))  # ceiling division

    for page in range(1, pages_needed + 1):
        if len(rows) >= count:
            break
        entries = fetch_page(page)
        if not entries:
            print(f"  No entries on page {page}, stopping.")
            break

        for entry in entries:
            if len(rows) >= count:
                break
            rid = entry.get("id", {}).get("label", "")
            if rid in seen_ids:
                continue
            seen_ids.add(rid)

            text = entry.get("content", {}).get("label", "")
            rating_raw = entry.get("im:rating", {}).get("label")
            rating = int(rating_raw) if rating_raw and rating_raw.isdigit() else None
            date_raw = entry.get("updated", {}).get("label", "")
            date_str = date_raw[:10] if date_raw else None

            extra = {
                "title": entry.get("title", {}).get("label", ""),
                "version": entry.get("im:version", {}).get("label", ""),
                "voteSum": entry.get("im:voteSum", {}).get("label"),
                "voteCount": entry.get("im:voteCount", {}).get("label"),
            }
            rows.append({
                "source": "appstore",
                "id": rid,
                "text": text,
                "rating": rating,
                "date": date_str,
                "extra_metadata": json.dumps(extra, ensure_ascii=False),
            })

        print(f"  Page {page}: {len(rows)} reviews collected so far.")

    df = pd.DataFrame(rows, columns=["source", "id", "text", "rating", "date", "extra_metadata"])
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} reviews to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=TARGET_COUNT)
    args = parser.parse_args()
    scrape(count=args.count)
