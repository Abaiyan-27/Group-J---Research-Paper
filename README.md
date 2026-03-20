# Group-J---Research-Paper

## User location split (country + other)

Use the script below to split `User_Location` into:
- `user_country` (sovereign country)
- `user_other` (state/region only; not city)

### Run (deterministic parser)

```bash
python scripts/split_user_location.py
```

Default input:
- `processed_tourism_reviews.csv`

Default output:
- `processed_tourism_reviews_country_other.csv`

### Optional LLM fallback for unresolved rows

Recommended: use a `.env` file.

1) Create `.env` from template:

```bash
cp .env.example .env
```

2) Edit `.env` with your provider values (`LLM_API_KEY`, `LLM_API_BASE_URL`, `LLM_MODEL`).

3) Run:

```bash
python scripts/split_user_location.py --use-llm --llm-model google/gemma-2-9b-it:free
```

The script auto-loads `.env`.

You can still use shell exports if preferred:

```bash
export LLM_API_KEY="your_provider_key"
export LLM_API_BASE_URL="https://openrouter.ai/api/v1"
python scripts/split_user_location.py --use-llm --llm-model google/gemma-2-9b-it:free
```

You can use other free providers by setting:

```bash
export LLM_API_KEY="your_provider_key"
export LLM_API_BASE_URL="https://your-provider.example/v1"
python scripts/split_user_location.py --use-llm --llm-model your-model-name
```

CLI flags are also supported:

```bash
python scripts/split_user_location.py --use-llm \
	--llm-api-key "your_provider_key" \
	--llm-api-base-url "https://your-provider.example/v1" \
	--llm-model "your-model-name"
```

The script appends these columns:
- `user_country`
- `user_other`
- `user_location_parse_method`
- `user_location_parse_confidence`

## Compare two LLM providers

Run two separate outputs (example: OpenRouter vs Groq), then compare them.

Recommended: set provider-specific variables in `.env`:
- `OPENROUTER_API_KEY`, `OPENROUTER_API_BASE_URL`, `OPENROUTER_MODEL`
- `GROQ_API_KEY`, `GROQ_API_BASE_URL`, `GROQ_MODEL`
- optional output controls: `OUTPUT_A_CSV`, `OUTPUT_B_CSV`, `MISMATCHES_OUT`

Then run everything in one command:

```bash
python scripts/run_provider_comparison.py
```

### Run A (OpenRouter example)

```bash
export LLM_API_KEY="your_openrouter_key"
export LLM_API_BASE_URL="https://openrouter.ai/api/v1"
python scripts/split_user_location.py --use-llm \
	--llm-model "google/gemma-2-9b-it:free" \
	--output-csv processed_tourism_reviews_country_other_openrouter.csv
```

### Run B (Groq example)

```bash
export LLM_API_KEY="your_groq_key"
export LLM_API_BASE_URL="https://api.groq.com/openai/v1"
python scripts/split_user_location.py --use-llm \
	--llm-model "llama-3.3-70b-versatile" \
	--output-csv processed_tourism_reviews_country_other_groq.csv
```

### Compare results

```bash
python scripts/compare_location_runs.py \
	--file-a processed_tourism_reviews_country_other_openrouter.csv \
	--file-b processed_tourism_reviews_country_other_groq.csv \
	--label-a openrouter \
	--label-b groq \
	--mismatches-out location_llm_mismatches.csv
```

This prints resolution rates and agreement metrics, and writes row-level differences to `location_llm_mismatches.csv`.