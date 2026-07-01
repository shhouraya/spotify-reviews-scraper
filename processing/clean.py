"""
Stage 2: Cleaning & Deduplication
Input:  analysis_outputs/raw_playstore.csv
        analysis_outputs/raw_appstore.csv
        analysis_outputs/raw_reddit.csv
Output: analysis_outputs/cleaned_dataset.csv

Steps:
  1. Merge raw CSVs into unified schema
  2. Drop empty / too-short entries (<5 words)
  3. Remove exact duplicates (hash on normalised text)
  4. Remove near-duplicates (Jaccard similarity on word sets, threshold 0.85)
  5. Flag non-English entries via langdetect (policy: keep, add lang column)
"""

import hashlib
import re
import unicodedata
import pandas as pd
from pathlib import Path

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not installed — language detection skipped.")

RAW_DIR = Path(__file__).parent.parent / "analysis_outputs"
OUTPUT_PATH = RAW_DIR / "cleaned_dataset.csv"

SOURCES = {
    "playstore": RAW_DIR / "raw_playstore.csv",
    "appstore":  RAW_DIR / "raw_appstore.csv",
    "reddit":    RAW_DIR / "raw_reddit.csv",
}

REQUIRED_COLUMNS = ["source", "id", "text", "rating", "date", "extra_metadata"]
MIN_WORDS = 5
NEAR_DUP_THRESHOLD = 0.85  # Jaccard similarity above this = near-duplicate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lowercase, strip punctuation/whitespace, normalise unicode."""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_hash(text: str) -> str:
    return hashlib.md5(normalise(text).encode("utf-8")).hexdigest()


def word_set(text: str) -> set[str]:
    return set(normalise(text).split())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def detect_lang(text: str) -> str:
    if not LANGDETECT_AVAILABLE:
        return "unknown"
    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"


def is_spam(text: str) -> bool:
    """Catch obvious spam patterns: all-caps, excessive repetition, URL-only."""
    t = text.strip()
    # Mostly caps (ignoring short texts)
    if len(t) > 20 and sum(1 for c in t if c.isupper()) / max(len(t), 1) > 0.8:
        return True
    # Single repeated character/word
    words = t.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return True
    # URL-only
    if re.fullmatch(r"https?://\S+", t):
        return True
    return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_and_merge() -> pd.DataFrame:
    frames = []
    for source, path in SOURCES.items():
        if not path.exists():
            print(f"  WARNING: {path.name} not found — skipping.")
            continue
        df = pd.read_csv(path, dtype=str)
        # Enforce unified schema; add missing cols as NaN
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[REQUIRED_COLUMNS]
        frames.append(df)
        print(f"  Loaded {len(df):>4} rows from {path.name}")
    return pd.concat(frames, ignore_index=True)


def remove_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    O(n²) Jaccard pass — acceptable at ≤1200 records.
    Keeps the first occurrence of any cluster of near-duplicates.
    """
    word_sets = [word_set(t) for t in df["text"]]
    keep = [True] * len(df)

    for i in range(len(df)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(df)):
            if not keep[j]:
                continue
            if jaccard(word_sets[i], word_sets[j]) >= NEAR_DUP_THRESHOLD:
                keep[j] = False

    dropped = keep.count(False)
    print(f"  Near-duplicate removal: dropped {dropped} rows (threshold={NEAR_DUP_THRESHOLD})")
    return df[keep].reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    initial = len(df)
    log = {}

    # 1. Drop rows with null/empty text
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    log["empty_text"] = initial - len(df)

    # 2. Drop too-short entries
    df = df[df["text"].apply(lambda t: len(str(t).split()) >= MIN_WORDS)]
    log["too_short"] = initial - log["empty_text"] - len(df)

    # 3. Drop spam
    pre_spam = len(df)
    df = df[~df["text"].apply(is_spam)]
    log["spam"] = pre_spam - len(df)

    # 4. Exact deduplication on normalised text hash
    df["_hash"] = df["text"].apply(text_hash)
    pre_exact = len(df)
    df = df.drop_duplicates(subset="_hash")
    log["exact_dupes"] = pre_exact - len(df)
    df = df.drop(columns=["_hash"])

    # 5. Near-duplicate removal
    pre_near = len(df)
    df = remove_near_duplicates(df)
    log["near_dupes"] = pre_near - len(df)

    # 6. Language detection — flag, don't drop
    print("  Running language detection (this may take a moment)...")
    df["lang"] = df["text"].apply(detect_lang)
    non_en = (df["lang"] != "en").sum()
    print(f"  Language detection: {non_en} non-English rows flagged (kept, labelled)")

    log["total_removed"] = initial - len(df)
    log["final_count"] = len(df)

    print(f"\n  Cleaning summary:")
    print(f"    Input rows:          {initial}")
    print(f"    Dropped (empty):     {log['empty_text']}")
    print(f"    Dropped (too short): {log['too_short']}")
    print(f"    Dropped (spam):      {log['spam']}")
    print(f"    Dropped (exact dup): {log['exact_dupes']}")
    print(f"    Dropped (near dup):  {log['near_dupes']}")
    print(f"    Non-English flagged: {non_en} (retained)")
    print(f"    Final rows:          {log['final_count']}")

    return df


def run():
    print("\n=== Stage 2: Cleaning & Deduplication ===\n")

    print("Loading raw files...")
    df = load_and_merge()
    print(f"  Total merged: {len(df)} rows\n")

    print("Cleaning...")
    df = clean(df)

    # Reorder: keep lang next to text for readability
    col_order = ["source", "id", "text", "lang", "rating", "date", "extra_metadata"]
    df = df[col_order]

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved cleaned dataset to {OUTPUT_PATH}")
    print("\n=== Done ===\n")
    return df


if __name__ == "__main__":
    run()
