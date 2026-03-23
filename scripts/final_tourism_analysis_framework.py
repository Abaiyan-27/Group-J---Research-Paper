#!/usr/bin/env python3
"""Final Tourism Review Analysis Framework.

Builds the full feature set requested in the final project specification and
exports an enriched dataset ready for analysis and reporting.
"""

from __future__ import annotations

import argparse
import importlib
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ----------------------------
# Utility and core transforms
# ----------------------------


def safe_to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def get_series(df: pd.DataFrame, column: str, default: object = "") -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if isinstance(value, (int, float, str, np.number)):
            return float(value)
        return default
    except (TypeError, ValueError):
        return default


def season_from_month(month: float) -> Optional[str]:
    if pd.isna(month):
        return None
    month_i = int(month)
    if month_i in (12, 1, 2):
        return "peak_dry"
    if month_i in (3, 4):
        return "inter_monsoon_1"
    if month_i in (5, 6, 7, 8, 9):
        return "southwest_monsoon"
    return "inter_monsoon_2"


def tourism_region_from_province(province: object) -> Optional[str]:
    if not isinstance(province, str) or not province.strip():
        return None

    key = province.strip().lower()
    mapping = {
        "western province": "west_coast_urban",
        "southern province": "south_coast",
        "eastern province": "east_coast",
        "northern province": "north_cultural",
        "central province": "hill_country",
        "uva province": "hill_country",
        "sabaragamuwa province": "rainforest_highlands",
        "north western province": "northwest_coast",
        "north central province": "cultural_triangle",
    }
    return mapping.get(key, "other_region")


def classify_rating(rating: float) -> str:
    if rating >= 4.0:
        return "positive"
    if rating >= 3.0:
        return "neutral"
    return "negative"


def classify_length(words: float) -> str:
    if words <= 50:
        return "short"
    if words <= 150:
        return "medium"
    return "long"


def reviewer_experience_level(contribs: float) -> str:
    if contribs < 5:
        return "low"
    if contribs < 50:
        return "medium"
    return "high"


def helpful_vote_bucket(votes: float) -> str:
    if votes <= 0:
        return "none"
    if votes <= 10:
        return "low"
    if votes <= 50:
        return "medium"
    return "high"


def sentiment_to_rating_anchor(label: str) -> int:
    if label == "positive":
        return 5
    if label == "negative":
        return 1
    return 3


def parse_dates(df: pd.DataFrame) -> None:
    df["Travel_Date"] = pd.to_datetime(
        get_series(df, "Travel_Date", ""), errors="coerce"
    )
    df["Published_Date"] = pd.to_datetime(
        get_series(df, "Published_Date", ""), errors="coerce", utc=True
    )

    df["travel_year"] = df["Travel_Date"].dt.year
    df["travel_month"] = df["Travel_Date"].dt.month
    df["travel_season"] = df["travel_month"].apply(season_from_month)

    df["published_year"] = df["Published_Date"].dt.year
    df["published_month"] = df["Published_Date"].dt.month

    delay = (df["Published_Date"].dt.tz_localize(None) - df["Travel_Date"]).dt.days
    df["review_delay_days"] = delay.clip(lower=0)


def build_descriptive_features(df: pd.DataFrame) -> None:
    df["review_count_per_location"] = (
        df.groupby("Location_Name")["Location_Name"].transform("count")
        if "Location_Name" in df.columns
        else np.nan
    )
    df["review_count_per_city"] = (
        df.groupby("Located_City")["Located_City"].transform("count")
        if "Located_City" in df.columns
        else np.nan
    )


def build_satisfaction_features(df: pd.DataFrame) -> None:
    rating = safe_to_numeric(get_series(df, "Rating", 0.0))
    df["Rating"] = rating

    if "Location_Name" in df.columns:
        df["avg_rating_location"] = df.groupby("Location_Name")["Rating"].transform(
            "mean"
        )
    else:
        df["avg_rating_location"] = np.nan

    if "Located_City" in df.columns:
        df["avg_rating_city"] = df.groupby("Located_City")["Rating"].transform("mean")
    else:
        df["avg_rating_city"] = np.nan

    df["rating_class"] = rating.apply(classify_rating)


def build_behavior_and_helpfulness(df: pd.DataFrame) -> None:
    text = get_series(df, "Text", "").fillna("").astype(str)
    title = get_series(df, "Title", "").fillna("").astype(str)

    df["review_length"] = text.str.len()
    df["word_count"] = text.str.split().str.len().fillna(0)
    df["title_length"] = title.str.len()

    contribs = safe_to_numeric(get_series(df, "User_Contributions", 0))
    votes = safe_to_numeric(get_series(df, "Helpful_Votes", 0))

    df["reviewer_experience_level"] = contribs.apply(reviewer_experience_level)
    df["helpfulness_ratio"] = (votes / (df["word_count"] + 1)).round(4)

    df["has_helpful_votes"] = (votes > 0).astype(int)
    df["helpful_vote_bucket"] = votes.apply(helpful_vote_bucket)

    df["length_bucket"] = df["word_count"].apply(classify_length)
    df["avg_helpful_by_length"] = df.groupby("length_bucket")[
        "Helpful_Votes"
    ].transform("mean")
    df["avg_rating_by_length"] = df.groupby("length_bucket")["Rating"].transform("mean")


def build_location_features(df: pd.DataFrame) -> None:
    if "province" not in df.columns:
        df["province"] = np.nan
    if "district" not in df.columns:
        df["district"] = np.nan

    df["tourism_region"] = df["province"].apply(tourism_region_from_province)


@dataclass
class ParseOriginResult:
    user_country: Optional[str]
    user_region: Optional[str]


def _country_aliases() -> Dict[str, str]:
    pycountry = importlib.import_module("pycountry")

    aliases: Dict[str, str] = {}
    for c in pycountry.countries:
        aliases[c.name.lower()] = c.name

    aliases.update(
        {
            "usa": "United States",
            "u.s.a": "United States",
            "us": "United States",
            "u.s": "United States",
            "uk": "United Kingdom",
            "u.k": "United Kingdom",
            "uae": "United Arab Emirates",
            "u.a.e": "United Arab Emirates",
            "russia": "Russia",
            "viet nam": "Vietnam",
            "korea, republic of": "South Korea",
        }
    )
    return aliases


COUNTRY_ALIASES = _country_aliases()

REGION_TO_COUNTRY = {
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "northern ireland": "United Kingdom",
    "new york": "United States",
    "california": "United States",
    "texas": "United States",
    "ontario": "Canada",
    "quebec": "Canada",
    "victoria": "Australia",
    "new south wales": "Australia",
}


def parse_user_origin(raw: object) -> ParseOriginResult:
    text = normalize_text(raw)
    if not text:
        return ParseOriginResult(user_country=None, user_region=None)

    parts = [p.strip() for p in text.split(",") if p.strip()]
    lower_parts = [p.lower() for p in parts]

    country = None
    for part in reversed(lower_parts):
        if part in COUNTRY_ALIASES:
            country = COUNTRY_ALIASES[part]
            break

    region = None
    for part in reversed(lower_parts):
        if part in REGION_TO_COUNTRY:
            region = part.title()
            if not country:
                country = REGION_TO_COUNTRY[part]
            break

    return ParseOriginResult(user_country=country, user_region=region)


def build_origin_features(df: pd.DataFrame) -> None:
    parsed = [
        parse_user_origin(x) for x in get_series(df, "User_Location", "").tolist()
    ]
    df["user_country"] = [x.user_country for x in parsed]
    df["user_region"] = [x.user_region for x in parsed]


# ----------------------------
# Sentiment + emotion
# ----------------------------


def load_transformer_pipelines(device: int, skip_models: set[str]):
    pipeline = importlib.import_module("transformers").pipeline
    models: Dict[str, object] = {}
    model_specs = {
        "cardiff": "cardiffnlp/twitter-roberta-base-sentiment",
        "siebert": "siebert/sentiment-roberta-large-english",
        "bertweet": "finiteautomata/bertweet-base-sentiment-analysis",
        "emotion": "j-hartmann/emotion-english-distilroberta-base",
    }

    for name, model_id in model_specs.items():
        if name in skip_models:
            print(f"Skipping model {name} (requested by --skip-models)")
            continue
        try:
            models[name] = pipeline(
                "text-classification",
                model=model_id,
                device=device,
            )
            print(f"Loaded model: {name} ({model_id})")
        except Exception as exc:
            print(f"Skipping model {name} ({model_id}) due to load error: {exc}")

    return models


def normalize_model_label(model_name: str, label: str) -> str:
    x = str(label).strip().upper()

    if model_name == "cardiff":
        if x in ("LABEL_2", "POSITIVE", "POS"):
            return "positive"
        if x in ("LABEL_0", "NEGATIVE", "NEG"):
            return "negative"
        return "neutral"

    if model_name == "siebert":
        if x in ("POSITIVE", "POS", "LABEL_1"):
            return "positive"
        return "negative"

    if model_name == "bertweet":
        if x in ("POS", "POSITIVE", "LABEL_2"):
            return "positive"
        if x in ("NEG", "NEGATIVE", "LABEL_0"):
            return "negative"
        return "neutral"

    return "neutral"


def batch_predict_texts(
    texts: Sequence[str],
    predictor,
    batch_size: int,
    model_name: str,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_index, i in enumerate(range(0, len(texts), batch_size), start=1):
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == total_batches:
            print(f"{model_name}: batch {batch_index}/{total_batches}")
        batch = [str(t)[:512] for t in texts[i : i + batch_size]]
        results = predictor(batch, truncation=True, batch_size=batch_size)
        for r in results:
            out.append({"label": r["label"], "score": float(r["score"])})
    return out


def compute_sentiment_columns(
    df: pd.DataFrame,
    batch_size: int,
    disable_models: bool,
    skip_models: set[str],
) -> None:
    combined = (
        get_series(df, "Title", "").fillna("").astype(str).str.strip()
        + " "
        + get_series(df, "Text", "").fillna("").astype(str).str.strip()
    ).str.strip()

    if disable_models:
        # Lightweight fallback if transformer inference is intentionally disabled.
        text_low = combined.str.lower()
        neg_words = text_low.str.count(
            r"\b(bad|poor|worst|dirty|crowded|disappointed)\b"
        )
        pos_words = text_low.str.count(
            r"\b(good|great|excellent|amazing|clean|friendly)\b"
        )
        score = (pos_words - neg_words).astype(float)

        df["combined_sentiment"] = np.where(
            score > 0, "positive", np.where(score < 0, "negative", "neutral")
        )
        df["sentiment_score"] = (score / (score.abs() + 1)).round(4)
        df["emotion"] = np.where(
            score > 0, "joy", np.where(score < 0, "sadness", "neutral")
        )
        return

    texts = combined.fillna("").tolist()

    torch = importlib.import_module("torch")
    device = 0 if torch.cuda.is_available() else -1
    print("Loading sentiment and emotion models...")
    models = load_transformer_pipelines(device, skip_models=skip_models)

    sentiment_model_order = ["cardiff", "siebert", "bertweet"]
    active_sentiment_models = [name for name in sentiment_model_order if name in models]

    if not active_sentiment_models:
        print(
            "No sentiment transformers available. Falling back to heuristic sentiment."
        )
        text_low = combined.str.lower()
        neg_words = text_low.str.count(
            r"\b(bad|poor|worst|dirty|crowded|disappointed)\b"
        )
        pos_words = text_low.str.count(
            r"\b(good|great|excellent|amazing|clean|friendly)\b"
        )
        score = (pos_words - neg_words).astype(float)
        df["combined_sentiment"] = np.where(
            score > 0, "positive", np.where(score < 0, "negative", "neutral")
        )
        df["sentiment_score"] = (score / (score.abs() + 1)).round(4)
        df["emotion"] = "neutral"
        return

    sentiment_predictions: Dict[str, List[Dict[str, object]]] = {}
    for name in active_sentiment_models:
        sentiment_predictions[name] = batch_predict_texts(
            texts, models[name], batch_size, model_name=name
        )

    emotion_raw: Optional[List[Dict[str, object]]] = None
    if "emotion" in models:
        emotion_raw = batch_predict_texts(
            texts, models["emotion"], batch_size, model_name="emotion"
        )

    final_labels: List[str] = []
    final_scores: List[float] = []
    emotions: List[str] = []

    for idx in range(len(texts)):
        label_to_vote = {"negative": -1, "neutral": 0, "positive": 1}
        votes: List[float] = []

        for model_name in active_sentiment_models:
            pred = sentiment_predictions[model_name][idx]
            pred_label = normalize_model_label(model_name, str(pred.get("label", "")))
            pred_score = to_float(pred.get("score", 0.0), 0.0)
            votes.append(label_to_vote[pred_label] * pred_score)

        mean_vote = float(np.mean(votes))
        if mean_vote > 0.1:
            final = "positive"
        elif mean_vote < -0.1:
            final = "negative"
        else:
            final = "neutral"

        final_labels.append(final)
        final_scores.append(round(mean_vote, 4))
        if emotion_raw is not None:
            emotions.append(str(emotion_raw[idx].get("label", "neutral")).lower())
        else:
            emotions.append("neutral")

    df["combined_sentiment"] = final_labels
    df["sentiment_score"] = final_scores
    df["emotion"] = emotions


def build_consistency_features(df: pd.DataFrame) -> None:
    sentiment_anchor = df["combined_sentiment"].map(
        {"positive": 1, "neutral": 0, "negative": -1}
    )
    rating_anchor = (df["Rating"] - 3.0) / 2.0

    # Positive means text is more positive than rating would suggest.
    df["sentiment_rating_gap"] = (sentiment_anchor - rating_anchor).round(4)

    sent_to_rating = df["combined_sentiment"].apply(sentiment_to_rating_anchor)
    rating_class = df["Rating"].apply(classify_rating)
    expected_sent = rating_class.map(
        {"positive": "positive", "neutral": "neutral", "negative": "negative"}
    )

    df["rating_sentiment_match"] = (df["combined_sentiment"] == expected_sent).astype(
        int
    )
    df["inconsistency_flag"] = (1 - df["rating_sentiment_match"]).astype(int)


def build_quality_score(df: pd.DataFrame) -> None:
    score = np.zeros(len(df), dtype=float)

    score += np.where(
        df["word_count"] > 150, 0.25, np.where(df["word_count"] > 60, 0.1, 0.0)
    )
    score += np.where(df["has_helpful_votes"] == 1, 0.2, 0.0)
    score += np.where(df["Helpful_Votes"] > 20, 0.15, 0.0)
    score += np.where(df["rating_sentiment_match"] == 1, 0.2, 0.0)
    score += np.clip(df["sentiment_score"].abs().to_numpy(), 0, 1) * 0.2

    df["review_quality_score"] = np.clip(score, 0, 1).round(4)


# ----------------------------
# Topic modeling
# ----------------------------


def infer_review_theme(topic_keywords: str, topic_id: int) -> str:
    words = topic_keywords.lower()

    theme_rules = [
        ("beach|sea|ocean|sand|surf", "beach_experience"),
        ("food|restaurant|dinner|breakfast|buffet", "food_and_dining"),
        ("hotel|room|staff|service|check", "hospitality_service"),
        ("temple|history|culture|museum", "culture_heritage"),
        ("wildlife|safari|park|elephant|nature", "nature_wildlife"),
        ("train|transport|road|bus|travel", "transport_access"),
        ("price|value|cost|money|expensive", "pricing_value"),
    ]

    for pattern, label in theme_rules:
        if re.search(pattern, words):
            return label

    if topic_id == -1:
        return "misc_outlier"
    return f"topic_{topic_id}"


def build_topic_features(
    df: pd.DataFrame, disable_topic: bool, min_topic_text_len: int
) -> None:
    df["dominant_topic"] = -1
    df["topic_probability"] = 0.0
    df["topic_keywords"] = ""
    df["review_theme"] = "misc_outlier"

    if disable_topic:
        print("Skipping BERTopic step (--disable-topic)")
        return

    BERTopic = importlib.import_module("bertopic").BERTopic

    text_for_topics = (
        get_series(df, "Title", "").fillna("").astype(str).str.strip()
        + " "
        + get_series(df, "Text", "").fillna("").astype(str).str.strip()
    ).str.strip()

    valid = df[text_for_topics.str.len() >= min_topic_text_len].copy()
    if valid.empty:
        print("Skipping BERTopic: no valid documents after length filter")
        return

    print(f"Running BERTopic on {len(valid)} reviews...")

    model = BERTopic(
        language="english",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        calculate_probabilities=True,
        min_topic_size=8,
        verbose=False,
    )

    docs = text_for_topics.loc[valid.index].tolist()
    topics, probs = model.fit_transform(docs)
    print("BERTopic completed")

    valid["dominant_topic"] = topics
    if probs is not None and len(probs) == len(valid):
        valid["topic_probability"] = [
            float(np.max(p)) if p is not None else 0.0 for p in probs
        ]
    else:
        valid["topic_probability"] = 0.0

    topic_to_keywords: Dict[int, str] = {}
    for topic_id in set(topics):
        if topic_id == -1:
            topic_to_keywords[topic_id] = ""
            continue
        words = model.get_topic(topic_id)
        topic_to_keywords[topic_id] = ", ".join([w for w, _ in (words or [])[:6]])

    valid["topic_keywords"] = valid["dominant_topic"].map(topic_to_keywords).fillna("")
    valid["review_theme"] = [
        infer_review_theme(kw, int(tid))
        for kw, tid in zip(valid["topic_keywords"], valid["dominant_topic"])
    ]

    df.loc[valid.index, "dominant_topic"] = valid["dominant_topic"].astype(int)
    df.loc[valid.index, "topic_probability"] = valid["topic_probability"].round(4)
    df.loc[valid.index, "topic_keywords"] = valid["topic_keywords"]
    df.loc[valid.index, "review_theme"] = valid["review_theme"]


# ----------------------------
# Destination performance
# ----------------------------


def build_destination_features(df: pd.DataFrame) -> None:
    destination_col = (
        "Location_Name" if "Location_Name" in df.columns else "Located_City"
    )

    sent_num = (
        df["combined_sentiment"]
        .map({"negative": 0.0, "neutral": 0.5, "positive": 1.0})
        .fillna(0.5)
    )
    df["destination_avg_rating"] = df.groupby(destination_col)["Rating"].transform(
        "mean"
    )
    df["destination_review_count"] = df.groupby(destination_col)[
        destination_col
    ].transform("count")
    df["destination_sentiment_mean"] = sent_num.groupby(df[destination_col]).transform(
        "mean"
    )

    rating_rank = (
        df.groupby(destination_col)["destination_avg_rating"]
        .first()
        .rank(ascending=False, method="dense")
    )
    review_rank = (
        df.groupby(destination_col)["destination_review_count"]
        .first()
        .rank(ascending=False, method="dense")
    )

    df["destination_rank_by_rating"] = (
        df[destination_col].map(rating_rank).astype("Int64")
    )
    df["destination_rank_by_reviews"] = (
        df[destination_col].map(review_rank).astype("Int64")
    )
    df["popularity_vs_quality_gap"] = (
        df["destination_rank_by_reviews"] - df["destination_rank_by_rating"]
    ).astype("Int64")


# ----------------------------
# Main
# ----------------------------


def run_pipeline(
    input_csv: str,
    output_csv: str,
    batch_size: int,
    disable_models: bool,
    disable_topic: bool,
    skip_models: set[str],
    min_topic_text_len: int,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv, encoding="utf-8", low_memory=False)
    print(f"Loaded input: {input_csv} ({len(df)} rows)")

    # Core standardization
    for col in ["Rating", "Helpful_Votes", "User_Contributions"]:
        if col in df.columns:
            df[col] = safe_to_numeric(df[col], default=0.0)

    build_descriptive_features(df)
    build_satisfaction_features(df)

    compute_sentiment_columns(
        df,
        batch_size=batch_size,
        disable_models=disable_models,
        skip_models=skip_models,
    )
    build_consistency_features(df)

    build_location_features(df)
    build_origin_features(df)
    parse_dates(df)

    build_behavior_and_helpfulness(df)
    build_quality_score(df)

    build_topic_features(
        df, disable_topic=disable_topic, min_topic_text_len=min_topic_text_len
    )
    build_destination_features(df)

    # Ensure expected framework columns exist, even under fallbacks.
    required_columns = [
        "review_count_per_location",
        "review_count_per_city",
        "avg_rating_location",
        "avg_rating_city",
        "rating_class",
        "combined_sentiment",
        "sentiment_score",
        "emotion",
        "sentiment_rating_gap",
        "dominant_topic",
        "topic_probability",
        "topic_keywords",
        "review_theme",
        "province",
        "district",
        "tourism_region",
        "user_country",
        "user_region",
        "travel_year",
        "travel_month",
        "travel_season",
        "published_year",
        "published_month",
        "review_delay_days",
        "review_length",
        "word_count",
        "title_length",
        "reviewer_experience_level",
        "helpfulness_ratio",
        "has_helpful_votes",
        "helpful_vote_bucket",
        "review_quality_score",
        "rating_sentiment_match",
        "inconsistency_flag",
        "destination_avg_rating",
        "destination_review_count",
        "destination_sentiment_mean",
        "length_bucket",
        "avg_helpful_by_length",
        "avg_rating_by_length",
        "destination_rank_by_rating",
        "destination_rank_by_reviews",
        "popularity_vs_quality_gap",
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan

    df.to_csv(output_csv, index=False, encoding="utf-8")
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the final tourism review analysis framework."
    )
    parser.add_argument(
        "--input-csv",
        default="processed_tourism_reviews_with_locations.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output-csv",
        default="processed_tourism_reviews_final_framework.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Batch size for transformer inference",
    )
    parser.add_argument(
        "--disable-models",
        action="store_true",
        help="Skip transformer sentiment/emotion models and use heuristic fallback.",
    )
    parser.add_argument(
        "--disable-topic",
        action="store_true",
        help="Skip BERTopic modeling.",
    )
    parser.add_argument(
        "--skip-models",
        nargs="*",
        default=[],
        help="Optional model names to skip: cardiff siebert bertweet emotion",
    )
    parser.add_argument(
        "--min-topic-text-len",
        type=int,
        default=30,
        help="Minimum text length for including reviews in BERTopic.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    valid_model_names = {"cardiff", "siebert", "bertweet", "emotion"}
    skip_models = {m.strip().lower() for m in args.skip_models}
    invalid = sorted(skip_models - valid_model_names)
    if invalid:
        raise ValueError(f"Unknown model names in --skip-models: {', '.join(invalid)}")

    df = run_pipeline(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        disable_models=args.disable_models,
        disable_topic=args.disable_topic,
        skip_models=skip_models,
        min_topic_text_len=args.min_topic_text_len,
    )

    print("Final framework complete")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
