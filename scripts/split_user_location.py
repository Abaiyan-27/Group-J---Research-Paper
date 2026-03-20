import argparse
import importlib
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from urllib import error, request

import pandas as pd
import pycountry


@dataclass
class ParseResult:
    country: Optional[str]
    other: Optional[str]
    method: str
    confidence: float


US_STATES = {
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
    "district of columbia",
    "puerto rico",
}

AU_STATES = {
    "new south wales",
    "queensland",
    "south australia",
    "tasmania",
    "victoria",
    "western australia",
    "australian capital territory",
    "northern territory",
}

CA_PROVINCES = {
    "alberta",
    "british columbia",
    "manitoba",
    "new brunswick",
    "newfoundland and labrador",
    "nova scotia",
    "ontario",
    "prince edward island",
    "quebec",
    "saskatchewan",
    "northwest territories",
    "nunavut",
    "yukon",
}

INDIA_STATES = {
    "andhra pradesh",
    "arunachal pradesh",
    "assam",
    "bihar",
    "chhattisgarh",
    "goa",
    "gujarat",
    "haryana",
    "himachal pradesh",
    "jharkhand",
    "karnataka",
    "kerala",
    "madhya pradesh",
    "maharashtra",
    "manipur",
    "meghalaya",
    "mizoram",
    "nagaland",
    "odisha",
    "punjab",
    "rajasthan",
    "sikkim",
    "tamil nadu",
    "telangana",
    "tripura",
    "uttar pradesh",
    "uttarakhand",
    "west bengal",
    "delhi",
    "jammu and kashmir",
    "ladakh",
}

UK_REGIONS = {"england", "scotland", "wales", "northern ireland"}


REGION_TO_COUNTRY = {
    **{state: "United States" for state in US_STATES},
    **{state: "Australia" for state in AU_STATES},
    **{state: "Canada" for state in CA_PROVINCES},
    **{state: "India" for state in INDIA_STATES},
    **{region: "United Kingdom" for region in UK_REGIONS},
    "new england": "United States",
}


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def clean_token(token: str) -> str:
    cleaned = re.sub(r"[^\w\s\-']", " ", token, flags=re.UNICODE)
    return normalize_space(cleaned)


def title_case_token(token: str) -> str:
    return " ".join(part.capitalize() for part in token.split())


def build_country_index() -> Tuple[Dict[str, str], Dict[str, str]]:
    country_alias_to_name: Dict[str, str] = {}

    for country in pycountry.countries:
        canonical = country.name
        country_alias_to_name[canonical.lower()] = canonical
        if hasattr(country, "official_name"):
            country_alias_to_name[country.official_name.lower()] = canonical
        if hasattr(country, "common_name"):
            country_alias_to_name[country.common_name.lower()] = canonical

    manual_aliases = {
        "usa": "United States",
        "u.s.a": "United States",
        "us": "United States",
        "u.s": "United States",
        "united states of america": "United States",
        "uk": "United Kingdom",
        "u.k": "United Kingdom",
        "england": "United Kingdom",
        "scotland": "United Kingdom",
        "wales": "United Kingdom",
        "northern ireland": "United Kingdom",
        "uae": "United Arab Emirates",
        "u.a.e": "United Arab Emirates",
        "the netherlands": "Netherlands",
        "holland": "Netherlands",
        "south korea": "Korea, Republic of",
        "north korea": "Korea, Democratic People's Republic of",
        "russia": "Russian Federation",
        "viet nam": "Vietnam",
    }
    country_alias_to_name.update(manual_aliases)

    country_short_to_preferred = {
        "Korea, Republic of": "South Korea",
        "Korea, Democratic People's Republic of": "North Korea",
        "Russian Federation": "Russia",
        "Viet Nam": "Vietnam",
        "Iran, Islamic Republic of": "Iran",
        "Syrian Arab Republic": "Syria",
        "Moldova, Republic of": "Moldova",
        "Tanzania, United Republic of": "Tanzania",
        "Venezuela, Bolivarian Republic of": "Venezuela",
        "Bolivia, Plurinational State of": "Bolivia",
        "Brunei Darussalam": "Brunei",
        "Lao People's Democratic Republic": "Laos",
        "Czechia": "Czech Republic",
        "United States": "United States",
        "United Kingdom": "United Kingdom",
    }

    return country_alias_to_name, country_short_to_preferred


COUNTRY_ALIAS_TO_NAME, COUNTRY_SHORT_TO_PREFERRED = build_country_index()


def to_preferred_country_name(canonical_name: str) -> str:
    return COUNTRY_SHORT_TO_PREFERRED.get(canonical_name, canonical_name)


def parse_user_location_deterministic(raw_value: object) -> ParseResult:
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return ParseResult(country=None, other=None, method="empty", confidence=0.0)

    raw_text = str(raw_value).strip()
    if not raw_text:
        return ParseResult(country=None, other=None, method="empty", confidence=0.0)

    parts = [clean_token(part) for part in raw_text.split(",")]
    parts = [part for part in parts if part]
    if not parts:
        return ParseResult(country=None, other=None, method="empty", confidence=0.0)

    country = None
    country_idx = None

    for idx in range(len(parts) - 1, -1, -1):
        token_lower = parts[idx].lower()
        if token_lower in COUNTRY_ALIAS_TO_NAME:
            canonical = COUNTRY_ALIAS_TO_NAME[token_lower]
            country = to_preferred_country_name(canonical)
            country_idx = idx
            break

    region_token = None
    region_country = None
    for idx in range(len(parts) - 1, -1, -1):
        token_lower = parts[idx].lower()
        if token_lower in REGION_TO_COUNTRY:
            region_token = title_case_token(token_lower)
            region_country = REGION_TO_COUNTRY[token_lower]
            if region_country == "United Kingdom" and token_lower in UK_REGIONS:
                region_token = token_lower.title()
            break

    if not country and region_country:
        country = region_country
        if region_token:
            return ParseResult(
                country=country,
                other=region_token,
                method="rule_region_infer_country",
                confidence=0.88,
            )
        return ParseResult(
            country=country,
            other=None,
            method="rule_region_infer_country",
            confidence=0.8,
        )

    if country:
        if region_token and REGION_TO_COUNTRY.get(region_token.lower()) == country:
            return ParseResult(
                country=country,
                other=region_token,
                method="rule_country_and_region",
                confidence=0.98,
            )

        if country_idx is not None:
            for idx in range(country_idx - 1, -1, -1):
                token_lower = parts[idx].lower()
                if token_lower in REGION_TO_COUNTRY and REGION_TO_COUNTRY[token_lower] == country:
                    return ParseResult(
                        country=country,
                        other=title_case_token(token_lower),
                        method="rule_country_plus_left_region",
                        confidence=0.95,
                    )

        return ParseResult(
            country=country,
            other=None,
            method="rule_country_only",
            confidence=0.9,
        )

    return ParseResult(country=None, other=None, method="unresolved", confidence=0.0)


def call_llm_for_location(
    raw_location: str,
    api_key: str,
    api_base_url: str,
    model: str,
    timeout_seconds: int,
) -> Optional[ParseResult]:
    prompt = (
        "Extract a sovereign country and a state/region value from this user location text. "
        "Rules: (1) other should be state/region only, never city. "
        "(2) map subregions to sovereign country, e.g., England->United Kingdom. "
        "(3) if unknown, use null. "
        "Return strict JSON only with keys: country, other, confidence.\n"
        f"location: {raw_location}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise location normalizer."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{api_base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            response_json = json.loads(resp.read().decode("utf-8"))
        content = response_json["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        country = parsed.get("country")
        other = parsed.get("other")
        confidence = float(parsed.get("confidence", 0.6))

        if country is not None and str(country).strip() == "":
            country = None
        if other is not None and str(other).strip() == "":
            other = None

        if country:
            normalized_country = COUNTRY_ALIAS_TO_NAME.get(str(country).lower(), country)
            country = to_preferred_country_name(normalized_country)

        if other:
            other_token = clean_token(str(other))
            if not other_token:
                other = None
            else:
                other = title_case_token(other_token.lower())

        return ParseResult(
            country=country,
            other=other,
            method="llm_fallback",
            confidence=max(0.0, min(confidence, 1.0)),
        )
    except (error.URLError, error.HTTPError, ValueError, KeyError, TypeError):
        return None


def run(
    input_csv: str,
    output_csv: str,
    location_column: str,
    use_llm: bool,
    llm_api_key: str,
    llm_api_base_url: str,
    llm_model: str,
    llm_timeout: int,
) -> None:
    df = pd.read_csv(input_csv, encoding="latin1")

    if location_column not in df.columns:
        raise ValueError(f"Column '{location_column}' not found in input CSV.")

    parsed = df[location_column].apply(parse_user_location_deterministic)
    df["user_country"] = parsed.apply(lambda x: x.country)
    df["user_other"] = parsed.apply(lambda x: x.other)
    df["user_location_parse_method"] = parsed.apply(lambda x: x.method)
    df["user_location_parse_confidence"] = parsed.apply(lambda x: x.confidence)

    llm_rows = 0
    api_key = llm_api_key.strip()

    if use_llm:
        if not api_key:
            print("LLM API key not found. Skipping LLM fallback.")
        else:
            unresolved_mask = df["user_country"].isna()
            unresolved_values = (
                df.loc[unresolved_mask, location_column].dropna().astype(str).str.strip()
            )
            unresolved_values = unresolved_values[unresolved_values != ""]

            unique_unresolved = unresolved_values.drop_duplicates().tolist()
            cache: Dict[str, ParseResult] = {}

            for raw_location in unique_unresolved:
                result = call_llm_for_location(
                    raw_location=raw_location,
                    api_key=api_key,
                    api_base_url=llm_api_base_url,
                    model=llm_model,
                    timeout_seconds=llm_timeout,
                )
                if result and result.country:
                    cache[raw_location] = result

            def apply_llm(row):
                nonlocal llm_rows
                current_country = row["user_country"]
                location_value = row[location_column]
                if pd.notna(current_country):
                    return row
                if pd.isna(location_value):
                    return row
                key = str(location_value).strip()
                result = cache.get(key)
                if not result:
                    return row
                row["user_country"] = result.country
                row["user_other"] = result.other
                row["user_location_parse_method"] = result.method
                row["user_location_parse_confidence"] = result.confidence
                llm_rows += 1
                return row

            df = df.apply(apply_llm, axis=1)

    output_dir = os.path.dirname(os.path.abspath(output_csv))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_csv, index=False)

    total_rows = len(df)
    resolved_rows = int(df["user_country"].notna().sum())
    unresolved_rows = total_rows - resolved_rows
    method_counts = df["user_location_parse_method"].value_counts(dropna=False)

    print(f"Input rows: {total_rows}")
    print(f"Resolved country rows: {resolved_rows}")
    print(f"Unresolved rows: {unresolved_rows}")
    print(f"Resolved ratio: {resolved_rows / total_rows:.2%}" if total_rows else "Resolved ratio: 0.00%")
    print(f"LLM-filled rows: {llm_rows}")
    print("Method counts:")
    print(method_counts)
    print(f"Wrote output: {output_csv}")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_dir, "..", ".env")
    try:
        dotenv_module = importlib.import_module("dotenv")
        load_dotenv = getattr(dotenv_module, "load_dotenv")
        load_dotenv(dotenv_path=env_path)
    except (ImportError, AttributeError):
        pass

    default_input = os.path.join(base_dir, "..", "processed_tourism_reviews.csv")
    default_output = os.path.join(
        base_dir,
        "..",
        "processed_tourism_reviews_country_other.csv",
    )

    parser = argparse.ArgumentParser(
        description="Split User_Location into sovereign country and state/region-only other.",
    )
    parser.add_argument(
        "--input-csv",
        default=default_input,
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-csv",
        default=default_output,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--location-column",
        default="User_Location",
        help="Column containing raw user location strings.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM fallback for unresolved rows (API key via --llm-api-key or env).",
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.getenv("LLM_API_KEY", ""),
        help="LLM API key from your selected free provider.",
    )
    parser.add_argument(
        "--llm-api-base-url",
        default=os.getenv("LLM_API_BASE_URL", "https://openrouter.ai/api/v1"),
        help="Provider base URL for the chat completions API.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("LLM_MODEL", "google/gemma-2-9b-it:free"),
        help="Model name used for LLM fallback.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=20,
        help="LLM request timeout in seconds.",
    )

    args = parser.parse_args()
    run(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        location_column=args.location_column,
        use_llm=args.use_llm,
        llm_api_key=args.llm_api_key,
        llm_api_base_url=args.llm_api_base_url,
        llm_model=args.llm_model,
        llm_timeout=args.llm_timeout,
    )


if __name__ == "__main__":
    main()
