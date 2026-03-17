import pandas as pd
import re
import os

# Resolve paths relative to this script so it works when invoked from any cwd
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_in = os.path.join(base_dir, "..", "Reviews.csv")
csv_out = os.path.join(base_dir, "..", "processed_tourism_reviews_with_locations.csv")

df = pd.read_csv(csv_in, encoding="latin1")

# Known Sri Lankan provinces (standard names)
provinces = [
    "Western Province",
    "Central Province",
    "Southern Province",
    "Northern Province",
    "Eastern Province",
    "North Western Province",
    "North Central Province",
    "Uva Province",
    "Sabaragamuwa Province",
]

# Small city -> district mapping (best-effort for common locations seen in dataset)
city_to_district = {
    "Arugam Bay": "Ampara",
    "Colombo": "Colombo",
    "Kandy": "Kandy",
    "Nuwara Eliya": "Nuwara Eliya",
    "Galle": "Galle",
    "Mirissa": "Matara",
    "Ella": "Badulla",
    "Negombo": "Gampaha",
    "Polonnaruwa": "Polonnaruwa",
    "Sigiriya": "Matale",
    "Trincomalee": "Trincomalee",
    "Jaffna": "Jaffna",
    "Batticaloa": "Batticaloa",
    "Anuradhapura": "Anuradhapura",
    "Matara": "Matara",
    "Kalutara": "Kalutara",
    "Bentota": "Galle",
    "Hikkaduwa": "Galle",
    "Mirigama": "Gampaha",
}

# Helpers
province_pattern = re.compile(
    r"(" + "|".join([re.escape(p) for p in provinces]) + r")", flags=re.IGNORECASE
)


def extract_province(location_str):
    if not isinstance(location_str, str) or not location_str.strip():
        return None
    # Try direct province match
    m = province_pattern.search(location_str)
    if m:
        # return canonical capitalization from list
        matched = m.group(1)
        # normalize to the one in provinces list (case-insensitive)
        for p in provinces:
            if p.lower() == matched.lower():
                return p
        return matched
    # If format like 'City, Province' try last token
    parts = [p.strip() for p in location_str.split(",") if p.strip()]
    if len(parts) >= 2:
        # assume last token is province or region
        return parts[-1]
    return None


def infer_district(row):
    city = row.get("Located_City")
    location = row.get("Location")
    # 1) mapping
    if isinstance(city, str) and city in city_to_district:
        return city_to_district[city]
    # 2) try to extract from location string if it contains a district-like token
    if isinstance(location, str):
        # common patterns: 'City, Province' or 'City, District, Province'
        parts = [p.strip() for p in location.split(",") if p.strip()]
        # If 3 tokens, middle might be district
        if len(parts) >= 3:
            return parts[-2]
        # If 2 tokens, first token often is city; use Located_City or first token
        if len(parts) == 2:
            first = parts[0]
            # if first looks like a city and differs from Located_City, use first
            if city and isinstance(city, str) and city.lower() == first.lower():
                # can't infer district, return city_to_district if exists
                return city_to_district.get(city)
            # otherwise return first as district fallback
            return first
    # 3) fallback to Located_City
    if isinstance(city, str) and city.strip():
        return city
    return None


# Apply extraction
print("Extracting province and district (best-effort)...")

df["province"] = df["Location"].apply(extract_province)
df["district"] = df.apply(infer_district, axis=1)

# Post-process: Normalize province values like remove extra tokens
# Strip trailing/leading whitespace
df["province"] = (
    df["province"]
    .astype(object)
    .apply(lambda x: x.strip() if isinstance(x, str) else x)
)
df["district"] = (
    df["district"]
    .astype(object)
    .apply(lambda x: x.strip() if isinstance(x, str) else x)
)

# Report
print("Province value counts (top 10):")
print(df["province"].value_counts(dropna=False).head(20))
print("\nSample districts:")
print(df["district"].value_counts().head(20))

# Save processed file
print(f"Writing: {csv_out}")
df.to_csv(csv_out, index=False)
print("Done.")
