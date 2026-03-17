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
    "Habarana": "Anuradhapura",
}

# Manual mappings for special Location strings
manual_location_mappings = {
    "Udawalawe National Park": {
        "province": "Sabaragamuwa Province",
        "district": "Ratnapura",
    },
    "North Central Province": {
        "province": "North Central Province",
        "district": "Anuradhapura",
    },
}

# Helpers
province_pattern = re.compile(
    r"(" + "|".join([re.escape(p) for p in provinces]) + r")", flags=re.IGNORECASE
)


def _normalize_text(value):
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _canonical_province_from_token(token):
    if not token:
        return None
    token_l = token.lower().strip()
    for p in provinces:
        p_l = p.lower()
        p_short = p_l.replace(" province", "")
        if token_l == p_l or token_l == p_short:
            return p
    return None


def _extract_parts(text):
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part and part.strip()]


def extract_province_from_row(row):
    # Use all tourism location columns in order of confidence.
    location = _normalize_text(row.get("Location"))
    located_city = _normalize_text(row.get("Located_City"))
    location_name = _normalize_text(row.get("Location_Name"))
    location_type = _normalize_text(row.get("Location_Type"))

    if not any([location, located_city, location_name, location_type]):
        return None

    # manual mapping exact match (case-insensitive)
    if location:
        for k, v in manual_location_mappings.items():
            if k.lower() == location.lower():
                return v.get("province")

    # direct match using known province names
    for candidate in [location, location_name]:
        if not candidate:
            continue
        m = province_pattern.search(candidate)
        if m:
            canonical = _canonical_province_from_token(m.group(1))
            if canonical:
                return canonical

    # catch trailing 'Province' text
    for candidate in [location, location_name]:
        if not candidate:
            continue
        m2 = re.search(r"([A-Za-z ]+Province)\b", candidate, flags=re.IGNORECASE)
        if m2:
            canonical = _canonical_province_from_token(m2.group(1).strip())
            if canonical:
                return canonical

    # comma-separated: last token might be province name without 'Province'
    for candidate in [location, location_name]:
        parts = _extract_parts(candidate)
        if parts:
            canonical = _canonical_province_from_token(parts[-1])
            if canonical:
                return canonical

    # Map city to its province through inferred district as fallback.
    if located_city and located_city in city_to_district:
        city_district = city_to_district[located_city]
        district_to_province = {
            "Ampara": "Eastern Province",
            "Colombo": "Western Province",
            "Kandy": "Central Province",
            "Nuwara Eliya": "Central Province",
            "Galle": "Southern Province",
            "Matara": "Southern Province",
            "Badulla": "Uva Province",
            "Gampaha": "Western Province",
            "Polonnaruwa": "North Central Province",
            "Matale": "Central Province",
            "Trincomalee": "Eastern Province",
            "Jaffna": "Northern Province",
            "Batticaloa": "Eastern Province",
            "Anuradhapura": "North Central Province",
            "Kalutara": "Western Province",
            "Ratnapura": "Sabaragamuwa Province",
        }
        return district_to_province.get(city_district)

    # Location_Type is weak for province inference; retain for future extensions.
    _ = location_type
    return None


def infer_district(row):
    city = _normalize_text(row.get("Located_City"))
    location = _normalize_text(row.get("Location"))
    location_name = _normalize_text(row.get("Location_Name"))
    location_type = _normalize_text(row.get("Location_Type"))

    # manual mapping exact match
    if location:
        loc = location
        for k, v in manual_location_mappings.items():
            if k.lower() == loc.lower():
                return v.get("district")

    # prefer explicit city mapping
    if city and city in city_to_district:
        return city_to_district[city]

    # Parse district hints from Location and Location_Name.
    for candidate in [location, location_name]:
        if not candidate:
            continue
        parts = _extract_parts(candidate)
        # if any part explicitly mentions 'District', use it
        for part in parts:
            if "district" in part.lower():
                return re.sub(r"\bdistrict\b", "", part, flags=re.IGNORECASE).strip()
        # if parts are like 'Town, District, Province' -> pick middle as district
        if len(parts) >= 3:
            return parts[-2]
        # if len==2 and first is likely district/city, return first
        if len(parts) == 2:
            first = parts[0]
            if city and city.lower() == first.lower():
                return city_to_district.get(city) or first
            return first
        # fallback: if single token is not province, use it
        if len(parts) == 1:
            single = parts[0]
            if _canonical_province_from_token(single) or "province" in single.lower():
                continue
            return single

    # Use Location_Name as a final fallback if it directly equals a known city.
    if location_name and location_name in city_to_district:
        return city_to_district[location_name]

    # Location_Type is weak for district inference; retain for future extensions.
    _ = location_type

    if city:
        return city
    return None


# Apply extraction
print("Extracting province and district (best-effort)...")

df["province"] = df.apply(extract_province_from_row, axis=1)
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
