# Group-J---Research-Paper

Final implementation of the Tourism Review Analysis Framework is available as a
reproducible script:

- `scripts/final_tourism_analysis_framework.py`

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
