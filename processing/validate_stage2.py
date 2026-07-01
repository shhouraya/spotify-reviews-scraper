"""
Stage 2 validation — tests all four cases from system_architecture.md.
Injects synthetic test rows into the raw data, re-runs cleaning, then
verifies results, before restoring the original files.
"""

import hashlib, re, unicodedata, shutil, sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from clean import clean, load_and_merge, OUTPUT_PATH, REQUIRED_COLUMNS

RAW_DIR = Path(__file__).parent.parent / "analysis_outputs"
SOURCES = {
    "playstore": RAW_DIR / "raw_playstore.csv",
    "appstore":  RAW_DIR / "raw_appstore.csv",
    "reddit":    RAW_DIR / "raw_reddit.csv",
}


def fmt(status: str, detail: str = "") -> str:
    marker = {"PASS": "OK", "FAIL": "!!"}.get(status, "--")
    return f"[{marker}] {status}" + (f": {detail}" if detail else "")


def inject_test_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Append synthetic rows and return their IDs for later verification."""
    ids = {}

    rows = []

    # T1a: exact duplicate of first real row
    base = df.iloc[0].copy()
    base["id"] = "TEST_exact_dup"
    ids["exact_dup"] = "TEST_exact_dup"
    rows.append(base)

    # T1b: near-duplicate — pick a row with enough words, change only one word
    # so Jaccard stays well above 0.85 regardless of text length
    long_rows = df[df["text"].apply(lambda t: len(str(t).split()) >= 15)]
    near_base = long_rows.iloc[0] if len(long_rows) else df.iloc[1]
    near = near_base.copy()
    original_words = str(near["text"]).split()
    modified = original_words.copy()
    modified[-1] = "TESTWORD"   # change exactly one word
    near["text"] = " ".join(modified)
    near["id"] = "TEST_near_dup"
    ids["near_dup"] = "TEST_near_dup"
    rows.append(near)

    # T2a: blank text
    blank = pd.Series({c: None for c in REQUIRED_COLUMNS})
    blank["source"] = "playstore"; blank["id"] = "TEST_blank"; blank["text"] = ""
    ids["blank"] = "TEST_blank"
    rows.append(blank)

    # T2b: too short (3 words)
    short = pd.Series({c: None for c in REQUIRED_COLUMNS})
    short["source"] = "playstore"; short["id"] = "TEST_short"; short["text"] = "ok good app"
    ids["short"] = "TEST_short"
    rows.append(short)

    # T2c: spam (single repeated word)
    spam = pd.Series({c: None for c in REQUIRED_COLUMNS})
    spam["source"] = "playstore"; spam["id"] = "TEST_spam"
    spam["text"] = "download download download download download download"
    ids["spam"] = "TEST_spam"
    rows.append(spam)

    # T3: non-English (French)
    foreign = pd.Series({c: None for c in REQUIRED_COLUMNS})
    foreign["source"] = "playstore"; foreign["id"] = "TEST_foreign"
    foreign["text"] = "Cette application est absolument magnifique pour decouvrir la musique"
    ids["foreign"] = "TEST_foreign"
    rows.append(foreign)

    injected = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return injected, ids


def run():
    print("\n=== Stage 2 Validation ===\n")

    # Load real merged data (don't modify files on disk)
    print("Loading real merged dataset...")
    real_df = load_and_merge()

    print("Injecting synthetic test rows...")
    test_df, ids = inject_test_rows(real_df)

    print("Running cleaning pipeline on augmented dataset...\n")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cleaned = clean(test_df)
    print(buf.getvalue())

    cleaned_ids = set(cleaned["id"].astype(str))

    print("--- Validation Results ---\n")

    # T1: Duplicates removed
    exact_gone = ids["exact_dup"] not in cleaned_ids
    near_gone  = ids["near_dup"]  not in cleaned_ids
    if exact_gone and near_gone:
        print(f"T1 Duplicates removed:   {fmt('PASS', 'exact dup gone, near-dup gone')}")
    else:
        detail = []
        if not exact_gone: detail.append("exact dup still present")
        if not near_gone:  detail.append("near dup still present")
        print(f"T1 Duplicates removed:   {fmt('FAIL', '; '.join(detail))}")

    # T2: Empty / spam filtered
    blank_gone = ids["blank"] not in cleaned_ids
    short_gone = ids["short"] not in cleaned_ids
    spam_gone  = ids["spam"]  not in cleaned_ids
    if blank_gone and short_gone and spam_gone:
        print(f"T2 Empty/spam filtered:  {fmt('PASS', 'blank, too-short, and spam rows all removed')}")
    else:
        detail = []
        if not blank_gone: detail.append("blank row still present")
        if not short_gone: detail.append("short row still present")
        if not spam_gone:  detail.append("spam row still present")
        print(f"T2 Empty/spam filtered:  {fmt('FAIL', '; '.join(detail))}")

    # T3: Schema integrity
    expected_cols = ["source", "id", "text", "lang", "rating", "date", "extra_metadata"]
    missing_cols = [c for c in expected_cols if c not in cleaned.columns]
    null_in_required = {c: cleaned[c].isna().sum() for c in ["source", "id", "text"]}
    bad_nulls = {k: v for k, v in null_in_required.items() if v > 0}
    if not missing_cols and not bad_nulls:
        print(f"T3 Schema integrity:     {fmt('PASS', f'all {len(expected_cols)} columns present, no nulls in required fields')}")
    else:
        detail = []
        if missing_cols: detail.append(f"missing cols: {missing_cols}")
        if bad_nulls:    detail.append(f"nulls in: {bad_nulls}")
        print(f"T3 Schema integrity:     {fmt('FAIL', '; '.join(detail))}")

    # T4: Non-English handling
    foreign_present = ids["foreign"] in cleaned_ids
    if foreign_present:
        lang_val = cleaned.loc[cleaned["id"] == ids["foreign"], "lang"].values
        lang_str = lang_val[0] if len(lang_val) else "?"
        print(f"T4 Non-English handling: {fmt('PASS', f'row kept with lang={lang_str!r} (policy: flag and retain)')}")
    else:
        print(f"T4 Non-English handling: {fmt('FAIL', 'non-English row was dropped (expected: kept with lang flag)')}")

    print(f"\n  Final cleaned rows (incl. injected non-dup rows): {len(cleaned)}")
    print("\n=== Done ===\n")


if __name__ == "__main__":
    run()
