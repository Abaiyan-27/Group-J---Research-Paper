# Group-J---Research-Paper

Final implementation of the Tourism Review Analysis Framework is available as a
reproducible script:

- `scripts/final_tourism_analysis_framework.py`

## Notebook 1 Reproducibility Update

`Tourism Analysis 1 - Dataset Preparation.ipynb` was hardened for reproducible runs.

What was changed:

- Removed runtime `pip install` from notebook cells.
- Added explicit dependency pinning in `requirements.txt`.
- Added `geopy` to requirements (previously used but not declared).
- Switched user-country resolution to offline-first mode by default:
  - direct alias matching + local cache (`country_resolution_cache.csv`)
  - optional live geocoding exists but is disabled by default
- Added fail-fast required-column checks before export.

How to run Notebook 1 reproducibly:

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Open `Tourism Analysis 1 - Dataset Preparation.ipynb` and run all cells from top.

3. Keep `ENABLE_GEOCODING = False` in the country-resolution cell for deterministic,
   network-independent runs.

4. Use generated timestamped output CSV for downstream notebooks/scripts.

## Notebook 2 Reproducibility Update

`Tourism Analysis 2 - Model Feature Generation.ipynb` was updated for stricter reproducibility on Kaggle.

What was changed:

- Pinned model revision for sentiment inference:
  - model: `cardiffnlp/twitter-roberta-base-sentiment`
  - revision: `daefdd1f6ae931839bce4d0f3db0a1a4265cd50f`
- Removed runtime package installation from the execution path (`datasets` must be preinstalled).
- Kept deterministic preprocessing and label mapping (`Title + Text` -> sentiment labels).
- Cleaned notebook execution logic to keep only required output generation (`Sentiment` column + CSV export).

Kaggle reproducibility evidence:

- Re-runs produced the same sentiment distribution:
  - `POSITIVE: 13041`, `NEGATIVE: 1602`, `NEUTRAL: 1513`
- Re-runs produced the same output hash:
  - `63e000b1328c36dee27c5f521bf6b5582f869891f7c3479a1b08d8090ecbeeb3`

How to run Notebook 2 reproducibly:

1. Use Kaggle Notebook with GPU enabled (T4 is sufficient).
2. Keep the same input dataset version in Kaggle.
3. Run all cells from top.
4. Archive `Processed_Reviews_with_Sentiment.csv` after each run.
5. Compare output file SHA256 between reruns; matching hashes indicate deterministic output for the same input and environment.

## Setup

```bash
python -m pip install -r requirements.txt
```

If you are on Python 3.14, `gensim` is skipped automatically because stable wheels
are not available yet for that version.

If you want a lighter install for quick testing (no transformer models), use:

```bash
python -m pip install pandas numpy pycountry scikit-learn python-dotenv
```

## Run Final Framework

```bash
python scripts/final_tourism_analysis_framework.py \
  --input-csv processed_tourism_reviews_with_locations.csv \
  --output-csv processed_tourism_reviews_final_framework.csv
```

Optional flags:

```bash
# Faster fallback (no transformer inference)
python scripts/final_tourism_analysis_framework.py --disable-models

# Skip BERTopic step
python scripts/final_tourism_analysis_framework.py --disable-topic
```

## Models Used (Final)

Sentiment ensemble:

- `cardiffnlp/twitter-roberta-base-sentiment`
- `siebert/sentiment-roberta-large-english`
- `finiteautomata/bertweet-base-sentiment-analysis`

Emotion model:

- `j-hartmann/emotion-english-distilroberta-base`

Topic modeling:

- `BERTopic`
- embedding model: `sentence-transformers/all-MiniLM-L6-v2`

Keyword extraction support:

- `KeyBERT` with `sentence-transformers/all-MiniLM-L6-v2` (supporting topic interpretation)

## Output Columns Added

The script generates the framework columns below:

- `review_count_per_location`, `review_count_per_city`
- `avg_rating_location`, `avg_rating_city`, `rating_class`
- `combined_sentiment`, `sentiment_score`, `emotion`, `sentiment_rating_gap`
- `dominant_topic`, `topic_probability`, `topic_keywords`, `review_theme`
- `province`, `district`, `tourism_region`
- `user_country`, `user_region`
- `travel_year`, `travel_month`, `travel_season`, `published_year`, `published_month`, `review_delay_days`
- `review_length`, `word_count`, `title_length`, `reviewer_experience_level`, `helpfulness_ratio`
- `has_helpful_votes`, `helpful_vote_bucket`, `review_quality_score`
- `rating_sentiment_match`, `inconsistency_flag`
- `destination_avg_rating`, `destination_review_count`, `destination_sentiment_mean`
- `length_bucket`, `avg_helpful_by_length`, `avg_rating_by_length`
- `destination_rank_by_rating`, `destination_rank_by_reviews`, `popularity_vs_quality_gap`
