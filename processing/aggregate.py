"""
Stage 4: Clustering & Aggregation
Input:  analysis_outputs/analyzed_dataset.json
Output: analysis_outputs/theme_summary.json

Approach:
- pain_point and user_goal labels are free-form synonyms, so we embed them
  with sentence-transformers and cluster with KMeans, then label each cluster.
- segment_signal variants are normalized with simple rules before aggregating.
- All percentages use field-specific denominators (only records where the field
  is non-null), per the spec.
"""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

INPUT_PATH  = Path(__file__).parent.parent / "analysis_outputs" / "analyzed_dataset.json"
OUTPUT_PATH = Path(__file__).parent.parent / "analysis_outputs" / "theme_summary.json"

# Number of clusters for each field — tuned to the data size and inspection
N_PAIN_CLUSTERS    = 10
N_GOAL_CLUSTERS    = 8
QUOTES_PER_THEME   = 3

# Manual overrides applied after clustering — fixes misleading auto-generated names.
# Key: substring to match in the auto-generated theme name (case-insensitive).
# Value: replacement display name.
PAIN_THEME_OVERRIDES = {
    "spotify not working":          "app reliability issues",
    "download doesn't work":        "app bugs and broken features",
    "ads during skips":             "free-tier restrictions",
    "auto-renewal issues":          "app bugs and broken features",
    "ai algo fails":                "AI and algorithm dissatisfaction",
    "disorganized playlists":       "playlist management issues",
    "cost of commercial free music":"subscription pricing complaints",
}
GOAL_THEME_OVERRIDES = {
    "sad songs":                    "emotion/mood-driven music seeking",
    "find music to ease depression":"emotion/mood-driven music seeking",
    "background noise while gaming":"ambient / background listening",
    "make mix sound good":          "audio quality and mix control",
    "expect default features":      "basic feature expectations",
}
MIN_QUOTE_WORDS    = 8   # skip very short quotes
MAX_QUOTE_CHARS    = 280


# ---------------------------------------------------------------------------
# Segment normalization (rule-based — variants are obvious)
# ---------------------------------------------------------------------------

SEGMENT_RULES = [
    # Tenure
    (r"long.?time|decade|years? (ago|back)|loyal|lifetime|veteran|old.?school|long.?term|former|used to", "long-time user"),
    # Subscription tier
    (r"free (tier|user|plan|trial|spotify)|non.?premium|no.?premium|without premium",  "free tier user"),
    (r"premium|paid|paying|subscri|pro user|pro plan",                                 "premium user"),
    # New users
    (r"new user|first time|just (started|installed|joined|got)|recent(ly)? (started|switched)", "new user"),
    # Roles / niche
    (r"podcast",                                      "podcast listener"),
    (r"dj|music producer|producer",                   "DJ / music producer"),
    (r"artist|musician|band|songwriter|performer",    "artist / musician"),
    (r"playlist (creator|maker|curator)|curator",     "playlist curator"),
    (r"developer|engineer",                           "developer"),
    # Demographics
    (r"student|college|university|school",            "student"),
    (r"senior|elderly|older (user|person|adult)",     "senior user"),
    (r"parent|kid|child|family",                      "parent / family user"),
    # Platform
    (r"android",                                      "Android user"),
    (r"iphone|ios|apple",                             "iOS user"),
    # Accessibility
    (r"adhd|autism|accessibility|disability|anxiety|depression",  "accessibility / mental health user"),
    # Music taste
    (r"(metal|rock|jazz|classical|hip.?hop|country|pop|edm|r&b|folk|indie|punk) (music |)fan", "genre enthusiast"),
    (r"music (enthusiast|fan|lover|nerd|obsessed|discovery|curator)|avid (listener|music)",    "music enthusiast"),
    (r"(discover|open to) new music|new music (fan|listener|seeker|enthusiast)",               "music discoverer"),
    # Catch-all for anything Spotify-specific
    (r"spotify (user|listener|fan|subscriber|executive|customer)",  "Spotify user"),
    # Generic heavy/casual
    (r"heavy user|daily user|power user|frequent",    "heavy user"),
    (r"casual (user|listener|gamer|player)",          "casual user"),
]

def normalize_segment(raw: str) -> str:
    s = raw.lower().strip()
    for pattern, label in SEGMENT_RULES:
        if re.search(pattern, s):
            return label
    return raw.strip()  # keep as-is if no rule matches


# ---------------------------------------------------------------------------
# Embedding + clustering
# ---------------------------------------------------------------------------

def embed(texts: list[str], model) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def best_k(embeddings: np.ndarray, k_min: int, k_max: int) -> int:
    """Pick k with highest silhouette score in range."""
    best, best_k_ = -1, k_min
    for k in range(k_min, min(k_max + 1, len(embeddings))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels, sample_size=min(500, len(embeddings)))
        if score > best:
            best, best_k_ = score, k
    return best_k_


def cluster_labels(raw_labels: list[str], n_clusters: int, model) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Returns ({raw_label: cluster_id}, embeddings_array, cluster_centers)."""
    unique = sorted(set(raw_labels))  # sorted for deterministic embedding order
    if len(unique) <= n_clusters:
        dummy_centers = np.zeros((len(unique), 1))
        return {lbl: i for i, lbl in enumerate(unique)}, dummy_centers, dummy_centers
    embs = embed(unique, model)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    ids = km.fit_predict(embs)
    return {lbl: int(ids[i]) for i, lbl in enumerate(unique)}, embs, km.cluster_centers_


def pick_cluster_name(labels_in_cluster: list[str], counts: Counter,
                      embs: np.ndarray, unique: list[str], center: np.ndarray) -> str:
    """Choose the most frequent raw label in the cluster as its display name."""
    return max(labels_in_cluster, key=lambda l: counts[l])


# ---------------------------------------------------------------------------
# Quote selection
# ---------------------------------------------------------------------------

def pick_quotes(records: list[dict], n: int = QUOTES_PER_THEME) -> list[str]:
    """Pick n diverse, readable quotes from a set of records."""
    candidates = [
        r["text"] for r in records
        if len(r["text"].split()) >= MIN_QUOTE_WORDS
    ]
    # Sort by length (prefer medium-length, not walls of text)
    candidates.sort(key=lambda t: abs(len(t) - 120))
    seen: set[str] = set()
    quotes = []
    for t in candidates:
        snippet = t[:MAX_QUOTE_CHARS].strip()
        if snippet not in seen:
            seen.add(snippet)
            quotes.append(snippet)
        if len(quotes) >= n:
            break
    return quotes


# ---------------------------------------------------------------------------
# Theme building
# ---------------------------------------------------------------------------

def build_themes(field: str, records: list[dict], label_to_cluster: dict[str, int],
                 cluster_names: dict[int, str], all_records_by_id: dict) -> list[dict]:
    """Build theme objects for one field (pain_point or user_goal)."""
    # Group record IDs by cluster
    cluster_records: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        val = r.get(field)
        if val is None:
            continue
        cid = label_to_cluster.get(val)
        if cid is None:
            continue
        cluster_records[cid].append(r)

    denominator = len(records)  # field-specific: only non-null records
    themes = []
    for cid, recs in sorted(cluster_records.items(), key=lambda x: -len(x[1])):
        # Segment breakdown for this theme
        seg_counts: Counter = Counter()
        for r in recs:
            seg = r.get("segment_signal")
            if seg:
                seg_counts[normalize_segment(seg)] += 1

        # Collect the raw label variants in this cluster
        raw_labels = list({r[field] for r in recs if r.get(field)})

        themes.append({
            "theme":             cluster_names[cid],
            "cluster_id":        cid,
            "count":             len(recs),
            "pct_of_field":      round(100 * len(recs) / denominator, 1),
            "example_quotes":    pick_quotes(recs),
            "segment_breakdown": dict(seg_counts.most_common()),
            "raw_labels":        sorted(raw_labels),
        })

    return themes


# ---------------------------------------------------------------------------
# Question mapping
# ---------------------------------------------------------------------------

def build_question_mapping(pain_themes: list[dict], goal_themes: list[dict],
                           segment_themes: list[dict], records: list[dict]) -> dict:
    """
    Map the six assignment questions to relevant themes + supporting stats.
    Matching is keyword-based against theme names.
    """

    def find_themes(theme_list: list[dict], keywords: list[str]) -> list[dict]:
        results = []
        for t in theme_list:
            name = t["theme"].lower()
            if any(kw in name for kw in keywords):
                results.append({"theme": t["theme"], "count": t["count"], "pct": t["pct_of_field"]})
        return results

    # Q4: repetition — cross-tab pain_point repetition themes with user goals
    repetition_records = [
        r for r in records
        if r.get("pain_point") and any(
            kw in (r["pain_point"] or "").lower()
            for kw in ["repeat", "same song", "same music", "stuck", "loop", "over and over"]
        )
    ]

    return {
        "q1_discovery_struggles": {
            "question": "Why do users struggle to discover new music?",
            "relevant_pain_themes": find_themes(pain_themes, ["discover", "recommendation", "algorithm", "new music", "suggest", "playlist", "ai"]),
            "relevant_goal_themes": find_themes(goal_themes, ["discover", "find new", "new music", "new songs", "new artist"]),
        },
        "q2_recommendation_frustrations": {
            "question": "What are the most common frustrations with recommendations?",
            "relevant_pain_themes": find_themes(pain_themes, ["recommend", "algorithm", "discover", "ai", "repeat", "same", "suggest", "playlist"]),
            "top_pain_themes_overall": [{"theme": t["theme"], "count": t["count"], "pct": t["pct_of_field"]} for t in pain_themes[:5]],
        },
        "q3_listening_behaviors": {
            "question": "What listening behaviors are users trying to achieve?",
            "relevant_goal_themes": goal_themes,  # all user goals are relevant here
        },
        "q4_repetition_causes": {
            "question": "What causes users to repeatedly listen to the same content?",
            "relevant_pain_themes": find_themes(pain_themes, ["repeat", "same", "algorithm", "shuffle", "recommend", "discover", "loop"]),
            "repetition_complaint_count": len(repetition_records),
            "example_quotes": pick_quotes(repetition_records, n=3),
        },
        "q5_segment_challenges": {
            "question": "Which user segments experience different discovery challenges?",
            "segment_themes": segment_themes,
            "discovery_pain_by_segment": _discovery_pain_by_segment(records),
        },
        "q6_unmet_needs": {
            "question": "What unmet needs emerge consistently across reviews?",
            "top_pain_themes": [{"theme": t["theme"], "count": t["count"], "pct": t["pct_of_field"]} for t in pain_themes[:8]],
            "top_goal_themes": [{"theme": t["theme"], "count": t["count"], "pct": t["pct_of_field"]} for t in goal_themes[:6]],
        },
    }


def _discovery_pain_by_segment(records: list[dict]) -> dict:
    discovery_kws = ["discover", "recommend", "algorithm", "new music", "suggest", "playlist", "repeat", "same song"]
    result: dict[str, dict] = {}
    for r in records:
        seg = r.get("segment_signal")
        if not seg:
            continue
        seg_norm = normalize_segment(seg)
        pp = (r.get("pain_point") or "").lower()
        has_discovery_pain = any(kw in pp for kw in discovery_kws)
        if seg_norm not in result:
            result[seg_norm] = {"total": 0, "discovery_pain": 0}
        result[seg_norm]["total"] += 1
        if has_discovery_pain:
            result[seg_norm]["discovery_pain"] += 1
    # Add percentage
    for seg, counts in result.items():
        counts["pct_with_discovery_pain"] = round(
            100 * counts["discovery_pain"] / counts["total"], 1
        ) if counts["total"] else 0
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("\n=== Stage 4: Clustering & Aggregation ===\n")

    with open(INPUT_PATH, encoding="utf-8") as f:
        records = json.load(f)

    print("Loading sentence-transformers model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- pain_point ---
    pp_records  = [r for r in records if r.get("pain_point")]
    pp_labels   = [r["pain_point"] for r in pp_records]
    pp_counts   = Counter(pp_labels)

    print(f"Clustering pain_point ({len(pp_records)} records, {len(set(pp_labels))} unique labels) → {N_PAIN_CLUSTERS} clusters...")
    pp_unique = list(set(pp_labels))
    pp_label_to_cluster, pp_embs, pp_centers = cluster_labels(pp_labels, N_PAIN_CLUSTERS, model)
    pp_cluster_members: dict[int, list[str]] = defaultdict(list)
    for lbl, cid in pp_label_to_cluster.items():
        pp_cluster_members[cid].append(lbl)
    pp_cluster_names = {
        cid: pick_cluster_name(lbls, pp_counts, pp_embs, pp_unique, pp_centers[cid])
        for cid, lbls in pp_cluster_members.items()
    }
    for cid, name in pp_cluster_names.items():
        for substring, replacement in PAIN_THEME_OVERRIDES.items():
            if substring.lower() in name.lower():
                pp_cluster_names[cid] = replacement
                print(f"  Renamed cluster {cid}: {name!r} -> {replacement!r}")

    pain_themes = build_themes("pain_point", pp_records, pp_label_to_cluster, pp_cluster_names, {})
    print(f"  -> {len(pain_themes)} pain point themes")

    # --- user_goal ---
    goal_records = [r for r in records if r.get("user_goal")]
    goal_labels  = [r["user_goal"] for r in goal_records]
    goal_counts  = Counter(goal_labels)

    print(f"Clustering user_goal ({len(goal_records)} records, {len(set(goal_labels))} unique labels) → {N_GOAL_CLUSTERS} clusters...")
    goal_unique = list(set(goal_labels))
    goal_label_to_cluster, goal_embs, goal_centers = cluster_labels(goal_labels, N_GOAL_CLUSTERS, model)
    goal_cluster_members: dict[int, list[str]] = defaultdict(list)
    for lbl, cid in goal_label_to_cluster.items():
        goal_cluster_members[cid].append(lbl)
    goal_cluster_names = {
        cid: pick_cluster_name(lbls, goal_counts, goal_embs, goal_unique, goal_centers[cid])
        for cid, lbls in goal_cluster_members.items()
    }
    # Apply manual name overrides
    for cid, name in goal_cluster_names.items():
        for substring, replacement in GOAL_THEME_OVERRIDES.items():
            if substring.lower() in name.lower():
                goal_cluster_names[cid] = replacement
                print(f"  Renamed cluster {cid}: {name!r} -> {replacement!r}")

    goal_themes = build_themes("user_goal", goal_records, goal_label_to_cluster, goal_cluster_names, {})
    print(f"  -> {len(goal_themes)} user goal themes")

    # --- segment_signal ---
    seg_records  = [r for r in records if r.get("segment_signal")]
    seg_norm_map = {r["segment_signal"]: normalize_segment(r["segment_signal"]) for r in seg_records}
    seg_counter  = Counter(seg_norm_map[r["segment_signal"]] for r in seg_records)

    # Build segment themes with example quotes per segment
    seg_records_by_norm: dict[str, list[dict]] = defaultdict(list)
    for r in seg_records:
        seg_records_by_norm[seg_norm_map[r["segment_signal"]]].append(r)

    segment_themes = []
    seg_denominator = len(seg_records)
    for seg_name, count in seg_counter.most_common():
        recs = seg_records_by_norm[seg_name]
        segment_themes.append({
            "segment":         seg_name,
            "count":           count,
            "pct_of_segments": round(100 * count / seg_denominator, 1),
            "example_quotes":  pick_quotes(recs),
            "raw_labels":      sorted({r["segment_signal"] for r in recs}),
        })
    print(f"  -> {len(segment_themes)} segment groups")

    # --- question mapping ---
    print("Building question mapping...")
    question_mapping = build_question_mapping(pain_themes, goal_themes, segment_themes, records)

    # --- assemble output ---
    summary = {
        "metadata": {
            "generated_at":              datetime.utcnow().isoformat() + "Z",
            "total_records":             len(records),
            "records_with_pain_point":   len(pp_records),
            "records_with_user_goal":    len(goal_records),
            "records_with_segment":      len(seg_records),
            "pain_point_clusters":       N_PAIN_CLUSTERS,
            "user_goal_clusters":        N_GOAL_CLUSTERS,
        },
        "pain_point_themes": pain_themes,
        "user_goal_themes":  goal_themes,
        "segment_themes":    segment_themes,
        "question_mapping":  question_mapping,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved theme_summary.json to {OUTPUT_PATH}")

    # Print a readable summary
    print("\n--- Pain Point Themes (by frequency) ---")
    for t in pain_themes:
        print(f"  {t['count']:3d} ({t['pct_of_field']:5.1f}%)  {t['theme']}")
        print(f"           variants: {', '.join(t['raw_labels'][:5])}")

    print("\n--- User Goal Themes ---")
    for t in goal_themes:
        print(f"  {t['count']:3d} ({t['pct_of_field']:5.1f}%)  {t['theme']}")

    print("\n--- Segment Groups ---")
    for s in segment_themes:
        print(f"  {s['count']:3d} ({s['pct_of_segments']:5.1f}%)  {s['segment']}")

    print("\n=== Done ===\n")


if __name__ == "__main__":
    run()
