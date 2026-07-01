"""
Stage 3: Automated Thematic Extraction via Groq API.
Input:  analysis_outputs/cleaned_dataset.csv
Output: analysis_outputs/analyzed_dataset.json

Processes one review at a time, writing results incrementally so a failure
mid-run doesn't lose completed work. Resumes from the last completed record
if re-run after an interruption.
"""

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIError

load_dotenv()

INPUT_PATH  = Path(__file__).parent.parent / "analysis_outputs" / "cleaned_dataset.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "analysis_outputs" / "analyzed_dataset.json"

MODEL = "llama-3.1-8b-instant"   # fast, free-tier, good at structured extraction
REQUEST_DELAY = 0.5               # seconds between requests (free tier: 30 RPM)
MAX_RETRIES = 6

SYSTEM_PROMPT = """You are a structured data extraction assistant. Your job is to read a user review of a music streaming app and extract exactly four fields as a JSON object.

Return ONLY a JSON object — no explanation, no markdown, no code fences. The object must have exactly these keys:

{
  "pain_point": <string or null>,
  "user_goal": <string or null>,
  "sentiment": <"positive" | "negative" | "neutral" | "mixed">,
  "segment_signal": <string or null>
}

Field definitions:
- "pain_point": A short category label (3-8 words) for the specific problem or frustration the reviewer describes. Set to null if the review expresses no clear complaint or frustration.
- "user_goal": A short phrase (3-8 words) describing what listening behaviour or outcome the user is trying to achieve. Set to null if no clear goal is expressed.
- "sentiment": The overall tone of the review. Must be one of: "positive", "negative", "neutral", "mixed". This field is always required — never null.
- "segment_signal": A short phrase (2-6 words) describing any identifiable characteristic of the user (e.g. "long-time subscriber", "free tier user", "new user", "podcast listener", "classical music fan"). Set to null if nothing is inferable from the text.

Critical rules:
- Do NOT force a value into pain_point, user_goal, or segment_signal if there is no clear textual evidence. Null is correct when the field does not apply.
- Do NOT invent or assume information not present in the review text.
- Keep all non-null values concise — short labels, not full sentences.
- sentiment must always be one of the four allowed values, never null."""

USER_PROMPT_TEMPLATE = """Review text:
\"\"\"
{text}
\"\"\""""


def build_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to your .env file.")
    return Groq(api_key=key)


def extract_json(content: str) -> dict | None:
    """Parse JSON from model response, stripping any accidental markdown."""
    content = content.strip()
    # Strip code fences if present
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to find a JSON object within the response
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


VALID_SENTIMENTS = {"positive", "negative", "neutral", "mixed"}

def validate(record: dict, row_id: str) -> dict:
    """Ensure schema compliance; fix or flag issues rather than crashing."""
    # Sentinel for extraction failure
    if record is None:
        return {
            "pain_point": None, "user_goal": None,
            "sentiment": "neutral", "segment_signal": None,
            "confidence_note": "extraction_failed",
        }

    # sentiment is mandatory
    sentiment = str(record.get("sentiment", "")).lower().strip()
    if sentiment not in VALID_SENTIMENTS:
        sentiment = "neutral"
        record["confidence_note"] = record.get("confidence_note", "") + " sentiment_coerced"

    return {
        "pain_point":     record.get("pain_point") or None,
        "user_goal":      record.get("user_goal") or None,
        "sentiment":      sentiment,
        "segment_signal": record.get("segment_signal") or None,
        "confidence_note": record.get("confidence_note") or None,
    }


def call_groq(client: Groq, text: str, row_id: str) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(text=text[:3000])},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            content = response.choices[0].message.content or ""
            parsed = extract_json(content)
            return validate(parsed, row_id)

        except RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"    Rate limited — waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
            time.sleep(wait)

        except APIError as e:
            wait = 5 * (attempt + 1)
            print(f"    API error ({e}) — retrying in {wait}s...")
            time.sleep(wait)

        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"    Unexpected error ({e}) — retrying in {wait}s...")
            time.sleep(wait)

    return validate(None, row_id)


def load_existing() -> dict[str, dict]:
    """Load already-completed records for resume support."""
    if not OUTPUT_PATH.exists():
        return {}
    try:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        return {r["id"]: r for r in existing}
    except Exception:
        return {}


def save(records: list[dict]) -> None:
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def run(limit: int | None = None, dry_run: bool = False) -> list[dict]:
    """
    limit: process only the first N rows (for test runs).
    dry_run: print what would be sent but make no API calls.
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = None if dry_run else build_client()

    df = pd.read_csv(INPUT_PATH, dtype=str)
    if limit:
        df = df.head(limit)

    existing = load_existing()
    completed_ids = set(existing.keys())
    records = list(existing.values())

    todo = df[~df["id"].isin(completed_ids)]
    total = len(df)
    done_at_start = len(completed_ids)

    print(f"\n=== Stage 3: Thematic Extraction ===")
    print(f"  Total rows:     {total}")
    print(f"  Already done:   {done_at_start}")
    print(f"  To process:     {len(todo)}")
    print(f"  Model:          {MODEL}")
    if dry_run:
        print("  [DRY RUN — no API calls]\n")

    for i, (_, row) in enumerate(todo.iterrows(), start=1):
        row_id = str(row["id"])
        text   = str(row["text"])

        if dry_run:
            print(f"  [{i}/{len(todo)}] Would extract: {row_id[:40]}  text[:60]={text[:60]!r}")
            continue

        extraction = call_groq(client, text, row_id)

        record = {
            "id":             row_id,
            "source":         str(row.get("source", "")),
            "text":           text,
            "rating":         row.get("rating") if pd.notna(row.get("rating")) else None,
            "date":           str(row.get("date", "")),
            "lang":           str(row.get("lang", "")),
            **extraction,
        }
        records.append(record)

        # Incremental save every 10 records
        if i % 10 == 0 or i == len(todo):
            save(records)
            pct = ((done_at_start + i) / total) * 100
            print(f"  [{done_at_start + i}/{total} | {pct:.0f}%] saved — last: {row_id[:40]}")

        time.sleep(REQUEST_DELAY)

    if not dry_run:
        save(records)
        print(f"\nExtraction complete. {len(records)} records saved to {OUTPUT_PATH}\n")

    return records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    parser.add_argument("--dry-run", action="store_true", help="Print rows without calling API")
    args = parser.parse_args()
    run(limit=args.limit, dry_run=args.dry_run)
