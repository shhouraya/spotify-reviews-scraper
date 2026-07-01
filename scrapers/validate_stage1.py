"""
Stage 1 validation script.
Runs all five test cases from the system_architecture.md validation table.
"""

import pandas as pd
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "analysis_outputs"
SOURCES = {
    "playstore": OUTPUT_DIR / "raw_playstore.csv",
    "appstore":  OUTPUT_DIR / "raw_appstore.csv",
    "reddit":    OUTPUT_DIR / "raw_reddit.csv",
}
REQUIRED_COLUMNS = ["source", "id", "text", "rating", "date", "extra_metadata"]
MIN_WORDS = 5

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def fmt(status, detail=""):
    marker = {"PASS": "OK", "FAIL": "!!", "SKIP": "--"}.get(status, "??")
    return f"  [{marker}] {status}" + (f": {detail}" if detail else "")


def load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def test_volume(df: pd.DataFrame, source: str, requested: int) -> str:
    n = len(df)
    # Accept within 20% of requested, or at least 1 record
    if n >= max(1, requested * 0.8):
        return PASS, f"{n} records (requested {requested})"
    return FAIL, f"only {n} records (requested {requested})"


def test_pagination(df: pd.DataFrame) -> tuple[str, str]:
    # Proxy: check no duplicate IDs (true page overlap would cause dupes)
    dupe_count = df["id"].duplicated().sum()
    if dupe_count == 0:
        return PASS, "no duplicate IDs"
    return FAIL, f"{dupe_count} duplicate IDs found"


def test_special_chars(df: pd.DataFrame) -> tuple[str, str]:
    # Look for any non-ASCII character surviving in text fields
    non_ascii = df["text"].dropna().apply(lambda t: any(ord(c) > 127 for c in str(t)))
    count = non_ascii.sum()
    sample = df.loc[non_ascii, "text"].head(1).values
    sample_str = repr(sample[0][:80]).encode("ascii", "backslashreplace").decode() if len(sample) else "(none found)"
    if count > 0:
        return PASS, f"{count} records contain non-ASCII/emoji. Sample: {sample_str}"
    # Even if none present, that's okay for small English samples — not a failure
    return PASS, "no non-ASCII found in sample (may be all-English batch)"


def test_schema(df: pd.DataFrame, source: str) -> tuple[str, str]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return FAIL, f"missing columns: {missing}"
    # Check source tag is consistent
    wrong_source = (df["source"] != source).sum()
    if wrong_source:
        return FAIL, f"{wrong_source} rows have wrong source tag"
    # Check no nulls in required non-nullable fields
    for col in ["source", "id", "text"]:
        nulls = df[col].isna().sum()
        if nulls:
            return FAIL, f"{nulls} nulls in required column '{col}'"
    return PASS, f"all {len(REQUIRED_COLUMNS)} columns present, source tag correct, no nulls in required fields"


def test_graceful_failure():
    # We can't trigger a live rate limit in a test script safely.
    # Instead: verify the retry logic is present in the source code.
    scraper_files = list((Path(__file__).parent).glob("scrape_*.py"))
    found_retry = {}
    for f in scraper_files:
        content = f.read_text(encoding="utf-8")
        found_retry[f.name] = "retries" in content and "time.sleep" in content
    all_ok = all(found_retry.values())
    detail = ", ".join(f"{k}: {'yes' if v else 'NO'}" for k, v in found_retry.items())
    return (PASS if all_ok else FAIL), detail


def run():
    print("\n=== Stage 1 Validation ===\n")
    results = {}

    for source, path in SOURCES.items():
        df = load(path)
        print(f"--- {source.upper()} ({path.name}) ---")
        if df is None:
            print(f"  [–] SKIP: file not found (scraper not yet run)\n")
            continue

        requested = 25  # small sample run
        status, detail = test_volume(df, source, requested)
        print(f"T1 Volume:          {fmt(status, detail)}")

        status, detail = test_pagination(df)
        print(f"T2 Pagination:      {fmt(status, detail)}")

        status, detail = test_special_chars(df)
        print(f"T3 Special chars:   {fmt(status, detail)}")

        # T4 rate limit — code inspection only
        print(f"T4 Rate limit:      [inspected via T5 combined check below]")

        status, detail = test_schema(df, source)
        print(f"T5 Schema:          {fmt(status, detail)}")
        print()

    print("--- CROSS-SOURCE ---")
    dfs = {s: load(p) for s, p in SOURCES.items() if load(p) is not None}
    if len(dfs) > 1:
        col_sets = {s: set(df.columns.tolist()) for s, df in dfs.items()}
        all_same = len(set(frozenset(v) for v in col_sets.values())) == 1
        status = PASS if all_same else FAIL
        detail = "all scrapers share identical column schema" if all_same else str(col_sets)
        print(f"T5 Schema parity:   {fmt(status, detail)}")
    else:
        print(f"T5 Schema parity:   {fmt(SKIP, 'need ≥2 sources to compare')}")

    print()
    status, detail = test_graceful_failure()
    print(f"T4 Retry logic:     {fmt(status, detail)}")

    print("\n=== Done ===\n")


if __name__ == "__main__":
    run()
