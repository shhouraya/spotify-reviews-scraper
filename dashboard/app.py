"""
Streamlit dashboard for AI-Powered Music App Review Analysis.
Pages 1-4 load from pre-computed JSON. Page 5 ("Ask the Data") calls
the Groq API live using the key stored in Streamlit secrets.
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Spotify Review Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent.parent / "analysis_outputs"


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_theme_summary() -> dict:
    with open(DATA_DIR / "theme_summary.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_analyzed_dataset() -> pd.DataFrame:
    with open(DATA_DIR / "analyzed_dataset.json", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Themes", "Question Answers", "Ask the Data", "Methodology & Limitations"],
)

summary = load_theme_summary()
df = load_analyzed_dataset()
meta = summary["metadata"]


# ---------------------------------------------------------------------------
# PAGE 1 — Overview
# ---------------------------------------------------------------------------

if page == "Overview":
    st.title("Spotify User Review Analysis")
    st.subheader("AI-Powered Analysis of Music Discovery & Recommendation Feedback")

    st.markdown(
        """
        This dashboard presents findings from an automated analysis of **{:,} Spotify user reviews**
        collected from the Google Play Store, Apple App Store, and Reddit. The goal is to surface
        patterns in how users experience music discovery and recommendations.
        """.format(meta["total_records"])
    )

    # Key metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{meta['total_records']:,}")
    c2.metric("Expressed a Pain Point", f"{meta['records_with_pain_point']:,}",
              f"{100*meta['records_with_pain_point']/meta['total_records']:.0f}% of reviews")
    c3.metric("Expressed a Goal", f"{meta['records_with_user_goal']:,}",
              f"{100*meta['records_with_user_goal']/meta['total_records']:.0f}% of reviews")
    c4.metric("Identified User Segment", f"{meta['records_with_segment']:,}",
              f"{100*meta['records_with_segment']/meta['total_records']:.0f}% of reviews")

    st.divider()

    # Source breakdown
    st.subheader("Data Sources")
    source_counts = df["source"].value_counts().reset_index()
    source_counts.columns = ["Source", "Reviews"]
    source_counts["Source"] = source_counts["Source"].map({
        "playstore": "Google Play Store",
        "appstore":  "Apple App Store",
        "reddit":    "Reddit",
    })

    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        st.bar_chart(source_counts.set_index("Source"))
    with col_table:
        st.dataframe(source_counts, use_container_width=True, hide_index=True)

    st.divider()

    # Sentiment breakdown
    st.subheader("Sentiment Distribution")
    sent_counts = df["sentiment"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment", "Count"]
    sent_order = ["positive", "negative", "neutral", "mixed"]
    sent_counts["Sentiment"] = pd.Categorical(sent_counts["Sentiment"], categories=sent_order, ordered=True)
    sent_counts = sent_counts.sort_values("Sentiment")

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        st.bar_chart(sent_counts.set_index("Sentiment"))
    with col_s2:
        st.dataframe(sent_counts, use_container_width=True, hide_index=True)

    st.divider()

    # Sample reviews
    st.subheader("Sample Reviews")
    st.caption("A random sample from the cleaned dataset.")
    sample = df[["source", "sentiment", "rating", "text"]].sample(min(8, len(df)), random_state=42)
    sample["source"] = sample["source"].map({
        "playstore": "Play Store",
        "appstore":  "App Store",
        "reddit":    "Reddit",
    })
    st.dataframe(sample, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE 2 — Themes
# ---------------------------------------------------------------------------

elif page == "Themes":
    st.title("Theme Analysis")
    st.markdown(
        "Extracted themes from {:,} reviews using the Groq API (Llama 3.1), then clustered "
        "with sentence-transformers embeddings and KMeans. Percentages use "
        "field-specific denominators — only reviews that expressed that field.".format(meta["total_records"])
    )

    tab1, tab2, tab3 = st.tabs(["Pain Points", "User Goals", "User Segments"])

    # --- Pain Points ---
    with tab1:
        st.subheader("Pain Point Themes")
        st.caption(
            f"{meta['records_with_pain_point']:,} of {meta['total_records']:,} reviews expressed a pain point "
            f"({100*meta['records_with_pain_point']/meta['total_records']:.0f}%). "
            "Percentages below are of that subset."
        )

        pp_themes = summary["pain_point_themes"]
        pp_df = pd.DataFrame([{
            "Theme": t["theme"].title(),
            "Count": t["count"],
            "% of Pain Point Reviews": t["pct_of_field"],
        } for t in pp_themes]).sort_values("Count", ascending=False)

        st.bar_chart(pp_df.set_index("Theme")["Count"])
        st.dataframe(pp_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Pain Point Details")
        selected_pp = st.selectbox(
            "Select a theme to see quotes and segment breakdown",
            [t["theme"].title() for t in pp_themes],
            key="pp_select",
        )
        pp_detail = next(t for t in pp_themes if t["theme"].title() == selected_pp)

        col_q, col_seg = st.columns(2)
        with col_q:
            st.markdown("**Example quotes:**")
            for q in pp_detail["example_quotes"]:
                st.markdown(f"> {q}")
        with col_seg:
            st.markdown("**User segments in this theme:**")
            if pp_detail["segment_breakdown"]:
                seg_df = pd.DataFrame(
                    pp_detail["segment_breakdown"].items(),
                    columns=["Segment", "Count"]
                ).sort_values("Count", ascending=False)
                st.dataframe(seg_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No segment data for this theme.")

    # --- User Goals ---
    with tab2:
        st.subheader("User Goal Themes")
        st.caption(
            f"{meta['records_with_user_goal']:,} of {meta['total_records']:,} reviews expressed a user goal "
            f"({100*meta['records_with_user_goal']/meta['total_records']:.0f}%). "
            "Percentages below are of that subset."
        )

        goal_themes = summary["user_goal_themes"]
        goal_df = pd.DataFrame([{
            "Theme": t["theme"].title(),
            "Count": t["count"],
            "% of Goal-Expressing Reviews": t["pct_of_field"],
        } for t in goal_themes]).sort_values("Count", ascending=False)

        st.bar_chart(goal_df.set_index("Theme")["Count"])
        st.dataframe(goal_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Goal Details")
        selected_goal = st.selectbox(
            "Select a theme to see quotes",
            [t["theme"].title() for t in goal_themes],
            key="goal_select",
        )
        goal_detail = next(t for t in goal_themes if t["theme"].title() == selected_goal)
        st.markdown("**Example quotes:**")
        for q in goal_detail["example_quotes"]:
            st.markdown(f"> {q}")

    # --- Segments ---
    with tab3:
        st.subheader("User Segments")
        st.caption(
            f"{meta['records_with_segment']:,} of {meta['total_records']:,} reviews contained an "
            f"identifiable user segment ({100*meta['records_with_segment']/meta['total_records']:.0f}%). "
            "Rare one-off segments (n=1) are shown collapsed below."
        )

        seg_themes = summary["segment_themes"]
        # Show only segments with count > 1 in the chart for readability
        sig_segs = [s for s in seg_themes if s["count"] > 1]
        seg_df = pd.DataFrame([{
            "Segment": s["segment"].title(),
            "Count": s["count"],
            "% of Segment-Tagged Reviews": s["pct_of_segments"],
        } for s in sig_segs]).sort_values("Count", ascending=False)

        st.bar_chart(seg_df.set_index("Segment")["Count"])
        st.dataframe(seg_df, use_container_width=True, hide_index=True)

        with st.expander("All segments including singletons"):
            all_seg_df = pd.DataFrame([{
                "Segment": s["segment"],
                "Count": s["count"],
            } for s in seg_themes])
            st.dataframe(all_seg_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE 3 — Question Answers
# ---------------------------------------------------------------------------

elif page == "Question Answers":
    st.title("Answers to the Six Research Questions")
    st.markdown(
        "Each answer is grounded directly in the extracted themes and review data. "
        "Supporting quotes are drawn from the analyzed dataset."
    )

    qm = summary["question_mapping"]

    def render_themes(theme_list: list[dict], label: str = "Relevant themes"):
        if not theme_list:
            st.caption("No directly matched themes.")
            return
        t_df = pd.DataFrame(theme_list)[["theme", "count", "pct_of_field"]].copy()
        t_df.columns = ["Theme", "Count", "% of field"]
        t_df["Theme"] = t_df["Theme"].str.title()
        st.caption(label)
        st.dataframe(t_df, use_container_width=True, hide_index=True)

    def render_quotes(quotes: list[str]):
        for q in quotes:
            st.markdown(f"> {q}")

    # Q1
    with st.expander("Q1 — Why do users struggle to discover new music?", expanded=True):
        q1 = qm["q1_discovery_struggles"]
        st.markdown(
            "**Finding**: Discovery struggles cluster around two pain points: the algorithm surfacing "
            "unwanted AI-generated content, and recommendations that feel repetitive or stagnant. "
            "Users want to find new music but feel the system works against them."
        )
        col1, col2 = st.columns(2)
        with col1:
            render_themes(q1["relevant_pain_themes"], "Pain points related to discovery")
        with col2:
            render_themes(q1["relevant_goal_themes"], "Discovery-related user goals")

    # Q2
    with st.expander("Q2 — What are the most common frustrations with recommendations?"):
        q2 = qm["q2_recommendation_frustrations"]
        st.markdown(
            "**Finding**: The top frustrations are repetitive shuffle algorithms, AI-generated music "
            "appearing in curated lists without clear labelling, and the inability to meaningfully "
            "influence what gets recommended."
        )
        render_themes(q2["relevant_pain_themes"], "Recommendation-related pain themes")
        st.divider()
        st.caption("Top pain themes overall (for context)")
        render_themes(q2["top_pain_themes_overall"])

    # Q3
    with st.expander("Q3 — What listening behaviors are users trying to achieve?"):
        q3 = qm["q3_listening_behaviors"]
        st.markdown(
            "**Finding**: The dominant behaviour is active music discovery — users want to find new "
            "artists and songs, not just replay familiar content. A significant secondary cluster "
            "seeks emotionally purposeful listening (mood, relaxation, background ambience)."
        )
        render_themes(q3["relevant_goal_themes"], "All user goal themes")

    # Q4
    with st.expander("Q4 — What causes users to repeatedly listen to the same content?"):
        q4 = qm["q4_repetition_causes"]
        st.markdown(
            f"**Finding**: {q4['repetition_complaint_count']} reviews explicitly mention repetition as a "
            "pain point. The primary causes are a shuffle algorithm that cycles through the same pool, "
            "recommendation playlists that stop updating, and AI-driven features that narrow variety "
            "rather than broadening it."
        )
        render_themes(q4["relevant_pain_themes"], "Repetition-related pain themes")
        if q4["example_quotes"]:
            st.markdown("**User quotes on repetition:**")
            render_quotes(q4["example_quotes"])

    # Q5
    with st.expander("Q5 — Which user segments experience different discovery challenges?"):
        q5 = qm["q5_segment_challenges"]
        st.markdown(
            "**Finding**: Long-time users (31% of segment-tagged reviews) are most vocal about "
            "algorithm regression — they recall Spotify working better and feel discovery has worsened. "
            "Free-tier users face structural discovery limits (skip caps, no on-demand). "
            "New users struggle with initial personalisation."
        )
        seg_data = q5["segment_themes"]
        sig = [s for s in seg_data if s["count"] > 1]
        if sig:
            s_df = pd.DataFrame([{
                "Segment": s["segment"].title(),
                "Count": s["count"],
                "% of Segments": s["pct_of_segments"],
            } for s in sig]).sort_values("Count", ascending=False)
            st.dataframe(s_df, use_container_width=True, hide_index=True)

        st.divider()
        st.caption("Discovery-related pain point rate by segment")
        disc_seg = q5["discovery_pain_by_segment"]
        if disc_seg:
            ds_df = pd.DataFrame([
                {"Segment": seg.title(), "Total Reviews": v["total"],
                 "With Discovery Pain": v["discovery_pain"],
                 "% With Discovery Pain": v["pct_with_discovery_pain"]}
                for seg, v in disc_seg.items()
                if v["total"] >= 3
            ]).sort_values("% With Discovery Pain", ascending=False)
            st.dataframe(ds_df, use_container_width=True, hide_index=True)

    # Q6
    with st.expander("Q6 — What unmet needs emerge consistently across reviews?"):
        q6 = qm["q6_unmet_needs"]
        st.markdown(
            "**Finding**: The clearest unmet needs are: (1) a recommendation engine that broadens "
            "rather than narrows listening; (2) meaningful discovery tools that surface genuinely new "
            "content; (3) reliable core app behaviour. These cut across all sources and segments."
        )
        col_p, col_g = st.columns(2)
        with col_p:
            render_themes(q6["top_pain_themes"], "Top pain themes (unmet needs)")
        with col_g:
            render_themes(q6["top_goal_themes"], "Top goal themes (desired outcomes)")


# ---------------------------------------------------------------------------
# PAGE 4 — Ask the Data
# ---------------------------------------------------------------------------

elif page == "Ask the Data":
    st.title("Ask the Data")
    st.markdown(
        "Ask any question about Spotify user reviews and get an AI-generated answer "
        "grounded in the **{:,} reviews** collected for this project. "
        "The model has access to all extracted themes, segment breakdowns, sentiment "
        "distributions, and example quotes — but not the raw review text.".format(meta["total_records"])
    )

    # Build the context string from theme_summary once (not inside the submit handler)
    def build_context(summary: dict) -> str:
        lines = []
        lines.append("=== DATASET OVERVIEW ===")
        m = summary["metadata"]
        lines.append(f"Total reviews: {m['total_records']}")
        lines.append(f"Sources: Google Play Store, Apple App Store, Reddit")
        lines.append(f"Reviews with a pain point: {m['records_with_pain_point']}")
        lines.append(f"Reviews with a user goal: {m['records_with_user_goal']}")
        lines.append(f"Reviews with a segment signal: {m['records_with_segment']}")

        lines.append("\n=== PAIN POINT THEMES ===")
        for t in summary["pain_point_themes"]:
            lines.append(
                f"- {t['theme']} (n={t['count']}, {t['pct_of_field']}% of pain-expressing reviews)"
            )
            if t.get("example_quotes"):
                lines.append(f"  Example: \"{t['example_quotes'][0]}\"")

        lines.append("\n=== USER GOAL THEMES ===")
        for t in summary["user_goal_themes"]:
            lines.append(
                f"- {t['theme']} (n={t['count']}, {t['pct_of_field']}% of goal-expressing reviews)"
            )
            if t.get("example_quotes"):
                lines.append(f"  Example: \"{t['example_quotes'][0]}\"")

        lines.append("\n=== USER SEGMENTS (count > 1) ===")
        for s in summary["segment_themes"]:
            if s["count"] > 1:
                lines.append(f"- {s['segment']} (n={s['count']}, {s['pct_of_segments']}% of segment-tagged reviews)")

        lines.append("\n=== SENTIMENT DISTRIBUTION ===")
        qm = summary["question_mapping"]
        # Pull sentiment from Q4 context which has total counts
        lines.append("(Derived from full dataset)")
        lines.append("Sentiments: positive, negative, neutral, mixed")

        lines.append("\n=== SEGMENT × DISCOVERY PAIN CROSSTAB ===")
        disc = qm.get("q5_segment_challenges", {}).get("discovery_pain_by_segment", {})
        for seg, v in disc.items():
            if v["total"] >= 3:
                lines.append(
                    f"- {seg}: {v['discovery_pain']}/{v['total']} reviews mention discovery pain "
                    f"({v['pct_with_discovery_pain']}%)"
                )

        return "\n".join(lines)

    SYSTEM_PROMPT = """You are an analyst assistant for a study of Spotify user reviews.
You have been given a structured summary of {n} reviews collected from the Google Play Store,
Apple App Store, and Reddit. The summary includes extracted pain point themes, user goal themes,
user segment breakdowns, and example quotes.

Answer the user's question using ONLY the data provided. Be specific — cite theme names, counts,
and percentages where relevant. Use bullet points for clarity when listing multiple findings.

If the question cannot be answered from the data (e.g. it asks about something not covered in
the reviews, or requires information outside the dataset), say so directly and briefly explain
what the data does and does not cover. Do not speculate or invent findings.

Keep answers concise: 3–6 sentences or a short bullet list. Do not repeat the question back."""

    context = build_context(summary)

    # Groq API key — try Streamlit secrets first, fall back to env var for local dev
    def get_groq_client():
        try:
            from groq import Groq
            api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
            if not api_key:
                return None, "GROQ_API_KEY not found. Add it to Streamlit secrets or your .env file."
            return Groq(api_key=api_key), None
        except ImportError:
            return None, "`groq` package not installed. Run `pip install groq`."

    # Suggested questions
    st.markdown("**Try a question, or write your own:**")
    suggestions = [
        "Which user segment is most frustrated with music discovery?",
        "What are the top unmet needs across all reviews?",
        "How do free-tier users differ from premium users in their complaints?",
        "What do users say about Spotify's shuffle algorithm?",
        "Which pain points are most common among long-time users?",
    ]
    cols = st.columns(len(suggestions))
    for i, (col, suggestion) in enumerate(zip(cols, suggestions)):
        if col.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
            st.session_state["ask_question_input"] = suggestion

    question = st.text_area(
        "Your question",
        value=st.session_state.get("ask_question_input", ""),
        placeholder="e.g. What are the biggest frustrations with Spotify's recommendation algorithm?",
        height=80,
        key="ask_question_input",
    )

    if st.button("Get Answer", type="primary", disabled=not question.strip()):
        client, err = get_groq_client()
        if err:
            st.error(err)
        else:
            with st.spinner("Analysing the reviews..."):
                try:
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT.format(n=meta["total_records"])
                                + "\n\n=== REVIEW DATA SUMMARY ===\n"
                                + context,
                            },
                            {"role": "user", "content": question.strip()},
                        ],
                        temperature=0.3,
                        max_tokens=600,
                    )
                    answer = response.choices[0].message.content.strip()
                    st.markdown("### Answer")
                    st.markdown(answer)
                    st.caption(
                        "Answer generated by Llama 3.1 (via Groq) from aggregated theme data. "
                        "Not based on individual review text."
                    )
                except Exception as e:
                    st.error(f"API error: {e}")

    st.divider()
    st.caption(
        "This feature uses the Groq API (Llama 3.1 8B). Answers are generated from "
        "pre-aggregated theme summaries, not the raw review text. Results may vary."
    )


# ---------------------------------------------------------------------------
# PAGE 5 — Methodology & Limitations
# ---------------------------------------------------------------------------

elif page == "Methodology & Limitations":
    st.title("Methodology & Limitations")

    st.subheader("Data Collection")
    st.markdown("""
- **Google Play Store**: scraped using `google-play-scraper` (Python), targeting the Spotify app (`com.spotify.music`), most recent English reviews.
- **Apple App Store**: scraped via the public iTunes RSS API (`itunes.apple.com/rss/customerreviews/`), no authentication required.
- **Reddit**: scraped via Reddit's public Atom/RSS listing feeds (`/r/spotify/top.rss`, etc.) across r/spotify, r/musicsuggest, and r/spotifyplaylist. Reddit's official Data API requires prior app approval incompatible with our timeline; the unauthenticated `.json` endpoints now return HTTP 403. RSS feeds are used as the publicly accessible alternative.
    """)

    st.subheader("Cleaning & Deduplication")
    st.markdown("""
- Merged all three sources into a unified schema.
- Removed reviews with fewer than 5 words, blank text, and obvious spam (repeated single word, URL-only).
- Exact deduplication via MD5 hash on normalised text; near-deduplication via Jaccard similarity (threshold 0.85).
- Non-English entries flagged using `langdetect` and retained (not dropped).
    """)

    st.subheader("Thematic Extraction")
    st.markdown(f"""
- Each review was processed by the **Groq API** (model: `llama-3.1-8b-instant`, free tier) using a structured extraction prompt.
- Extracted fields per review: `pain_point`, `user_goal`, `sentiment`, `segment_signal`.
- The model was explicitly instructed to return `null` for optional fields when no clear textual evidence exists — not to infer or force values.
- Validated against a 15-review hand-labelled sample before running the full batch.
- Total records extracted: **{meta['total_records']:,}** with 0 API failures.
    """)

    st.subheader("Clustering & Aggregation")
    st.markdown("""
- `pain_point` and `user_goal` labels are free-form (extracted by the LLM), so synonym grouping was performed using `sentence-transformers` (`all-MiniLM-L6-v2`) embeddings + KMeans clustering.
- 10 pain point clusters and 8 user goal clusters were used, tuned by inspection.
- Cluster names were chosen as the most-frequent raw label within each cluster, with manual overrides applied where the auto-name was misleading.
- All theme percentages use **field-specific denominators**: e.g. pain point percentages divide only by the count of reviews that expressed a pain point, not the full dataset.
    """)

    st.subheader("Limitations")
    st.markdown(f"""
1. **Sample, not population**: The dataset ({meta['total_records']:,} records) is a snapshot, not an exhaustive collection. Recent reviews are over-represented.
2. **LLM extraction error**: Thematic extraction used an open-weight model (Llama 3.1 8B) validated only against a 15-record hand-labelled sample. Some misclassification is expected, particularly for sarcasm and ambiguous short reviews.
3. **Reddit source constraints**: Reddit data comes from listing feeds only (no keyword search), so it skews toward popular/highly-upvoted posts rather than a random sample of user opinion.
4. **No inter-rater reliability**: Extraction accuracy is approximated by spot-checking, not formally measured.
5. **Static dashboard**: All results are pre-computed. The dashboard does not reflect new reviews posted after the data collection date.
    """)
