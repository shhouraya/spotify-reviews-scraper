"""
Play Store scraper for Spotify reviews.
Outputs: analysis_outputs/raw_playstore.csv
Columns: source, id, text, rating, date, extra_metadata
"""

import json
import time
import pandas as pd
from pathlib import Path
from google_play_scraper import reviews, Sort
from google_play_scraper.exceptions import NotFoundError

APP_ID = "com.spotify.music"
TARGET_COUNT = 400
OUTPUT_PATH = Path(__file__).parent.parent / "analysis_outputs" / "raw_playstore.csv"


def scrape(count: int = TARGET_COUNT) -> pd.DataFrame:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_reviews = []
    continuation_token = None
    retries = 0
    max_retries = 5

    print(f"Scraping {count} Play Store reviews for {APP_ID}...")

    while len(all_reviews) < count:
        batch_size = min(200, count - len(all_reviews))
        try:
            result, continuation_token = reviews(
                APP_ID,
                lang="en",
                country="us",
                sort=Sort.NEWEST,
                count=batch_size,
                continuation_token=continuation_token,
            )
        except NotFoundError:
            print(f"App {APP_ID} not found.")
            break
        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"Max retries exceeded: {e}")
                break
            wait = 2 ** retries
            print(f"Error ({e}), retrying in {wait}s...")
            time.sleep(wait)
            continue

        if not result:
            print("No more results returned.")
            break

        all_reviews.extend(result)
        retries = 0
        print(f"  Fetched {len(all_reviews)} so far...")

        if continuation_token is None:
            break

    rows = []
    seen_ids = set()
    for r in all_reviews:
        rid = str(r.get("reviewId", ""))
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        extra = {
            "thumbsUpCount": r.get("thumbsUpCount"),
            "reviewCreatedVersion": r.get("reviewCreatedVersion"),
            "replyContent": r.get("replyContent"),
        }
        rows.append({
            "source": "playstore",
            "id": rid,
            "text": r.get("content", ""),
            "rating": r.get("score"),
            "date": pd.to_datetime(r.get("at")).date().isoformat() if r.get("at") else None,
            "extra_metadata": json.dumps(extra, ensure_ascii=False),
        })

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
