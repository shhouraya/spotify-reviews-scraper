"""
Reddit scraper using Reddit's public Atom/RSS feeds (no auth, no API key).

WHY NOT THE OFFICIAL DATA API:
Reddit's current developer terms require explicit app approval before API access
is granted to new apps, and the "script" app type now triggers a Responsible
Builder Policy review gate. Both pathways block progress on a short deadline.
The unauthenticated .json endpoints are also now returning HTTP 403 for all
requests regardless of User-Agent.

This scraper therefore uses Reddit's public Atom/RSS listing endpoints
(e.g. /r/spotify/top.rss), which remain accessible without authentication
and return the same posts a logged-out visitor sees in their browser.
Use is read-only, non-commercial, and limited to ~300-400 records for an
academic assignment. No posting, voting, or account interaction.

Outputs: analysis_outputs/raw_reddit.csv
Columns: source, id, text, rating, date, extra_metadata
"""

import html
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "analysis_outputs" / "raw_reddit.csv"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

NS = {"atom": "http://www.w3.org/2005/Atom"}

# (subreddit, sort, time_filter)  — multiple combos to reach target volume
FEEDS = [
    ("spotify",        "top",  "year"),
    ("spotify",        "top",  "month"),
    ("spotify",        "hot",  ""),
    ("spotify",        "new",  ""),
    ("musicsuggest",   "top",  "year"),
    ("musicsuggest",   "top",  "month"),
    ("musicsuggest",   "hot",  ""),
    ("spotifyplaylist","top",  "year"),
    ("spotifyplaylist","hot",  ""),
]

REQUEST_DELAY = 3.0   # seconds between requests — conservative to avoid 429


def strip_html(raw: str) -> str:
    """Strip HTML tags and decode entities from RSS content."""
    text = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_feed(subreddit: str, sort: str, time_filter: str,
               after: str | None, retries: int = 5) -> tuple[list[ET.Element], str | None]:
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.rss?limit=100"
    if time_filter:
        url += f"&t={time_filter}"
    if after:
        url += f"&after={after}"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = resp.read()
            root = ET.fromstring(body)
            entries = root.findall("atom:entry", NS)

            # Derive the after-token from the last entry's fullname id (t3_xxxx)
            next_after = None
            if entries:
                last_id = entries[-1].findtext("atom:id", namespaces=NS) or ""
                # id looks like "t3_abc123"
                if re.match(r"t\d_\w+", last_id):
                    next_after = last_id

            return entries, next_after
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 15 * (attempt + 1)
                print(f"    429 rate-limited, waiting {wait}s...")
                time.sleep(wait)
            elif e.code in (403, 404):
                print(f"    HTTP {e.code} for {url} — skipping.")
                return [], None
            else:
                wait = 2 ** (attempt + 1)
                print(f"    HTTP {e.code}, retry in {wait}s...")
                time.sleep(wait)
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"    Error ({e}), retry in {wait}s...")
            time.sleep(wait)
    return [], None


def entry_to_row(entry: ET.Element, subreddit: str, sort: str) -> dict | None:
    fullname = entry.findtext("atom:id", namespaces=NS) or ""
    if not fullname:
        return None

    title = (entry.findtext("atom:title", namespaces=NS) or "").strip()
    content_el = entry.find("atom:content", NS)
    content_html = (content_el.text or "") if content_el is not None else ""
    body = strip_html(content_html).strip()

    # Combine title + body for maximum signal; body often repeats title so dedup
    if body and body.lower() != title.lower():
        text = f"{title}\n\n{body}"
    else:
        text = title

    if not text.strip():
        return None

    updated = entry.findtext("atom:updated", namespaces=NS) or ""
    date_str = updated[:10] if updated else None

    link_el = entry.find("atom:link", NS)
    link = link_el.get("href", "") if link_el is not None else ""

    author_el = entry.find("atom:author/atom:name", NS)
    author = author_el.text if author_el is not None else ""

    extra = {
        "title": title,
        "subreddit": subreddit,
        "sort": sort,
        "author": author,
        "link": link,
        "type": "post",
    }

    return {
        "source": "reddit",
        "id": fullname,
        "text": text,
        "rating": None,
        "date": date_str,
        "extra_metadata": json.dumps(extra, ensure_ascii=False),
    }


def scrape(count: int = 400) -> pd.DataFrame:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scraping up to {count} Reddit records via public RSS (no auth)...")

    all_rows: list[dict] = []
    seen_ids: set[str] = set()

    for subreddit, sort, time_filter in FEEDS:
        if len(all_rows) >= count:
            break

        label = f"r/{subreddit}/{sort}" + (f"?t={time_filter}" if time_filter else "")
        print(f"\n  {label}:")

        after = None
        page = 0
        feed_rows = 0

        while len(all_rows) < count:
            if page > 0:
                time.sleep(REQUEST_DELAY)

            entries, next_after = fetch_feed(subreddit, sort, time_filter, after)
            page += 1

            if not entries:
                break

            for entry in entries:
                row = entry_to_row(entry, subreddit, sort)
                if row and row["id"] not in seen_ids:
                    seen_ids.add(row["id"])
                    all_rows.append(row)
                    feed_rows += 1

            print(f"    page {page}: +{feed_rows} this feed, {len(all_rows)} total")

            # Only paginate if there's more to get and we got a full page
            if next_after and len(entries) == 100 and len(all_rows) < count:
                after = next_after
                time.sleep(REQUEST_DELAY)
            else:
                break

    df = pd.DataFrame(all_rows, columns=["source", "id", "text", "rating", "date", "extra_metadata"])
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} records to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=400)
    args = parser.parse_args()
    scrape(count=args.count)
