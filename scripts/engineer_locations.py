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
    'Habarana': 'Anuradhapura',
}

# Manual mappings for special Location strings
manual_location_mappings = {
    'Udawalawe National Park': {'province': 'Sabaragamuwa Province', 'district': 'Ratnapura'},
    'North Central Province': {'province': 'North Central Province', 'district': 'Anuradhapura'},
}

# Helpers
province_pattern = re.compile(
    r"(" + "|".join([re.escape(p) for p in provinces]) + r")", flags=re.IGNORECASE
)


def extract_province(location_str):
    if not isinstance(location_str, str) or not location_str.strip():
        return None
    loc = location_str.strip()
    # manual mapping exact match (case-insensitive)
    for k, v in manual_location_mappings.items():
        if k.lower() == loc.lower():
            return v.get('province')
    # direct match using known province names
    m = province_pattern.search(loc)
    if m:
        matched = m.group(1)
        for p in provinces:
            if p.lower() == matched.lower():
                return p
        return matched
    # catch trailing 'Province' text
    m2 = re.search(r"([A-Za-z ]+Province)\\b", loc)
    if m2:
        return m2.group(1).strip()
    # comma-separated: last token might be province name without 'Province'
    parts = [p.strip() for p in loc.split(',') if p.strip()]
    if parts:
        last = parts[-1]
        for p in provinces:
            if last.lower() == p.replace(' Province', '').lower() or last.lower() == p.lower():
                return p
    return None


def infer_district(row):
    city = row.get("Located_City")
    location = row.get("Location")
    # manual mapping exact match
    if isinstance(location, str):
        loc = location.strip()
        for k, v in manual_location_mappings.items():
            if k.lower() == loc.lower():
                return v.get('district')
    # prefer explicit city mapping
    if isinstance(city, str) and city in city_to_district:
        return city_to_district[city]
    if not isinstance(location, str) or not location.strip():
        return None
    parts = [p.strip() for p in location.split(',') if p.strip()]
    # if any part explicitly mentions 'District', use it
    for part in parts:
        if 'District' in part:
            return part.replace('District', '').strip()
    # if location is just a province name, try to use city as district
    if len(parts) == 1 and (any(parts[0].lower() == p.lower() for p in provinces) or 'province' in parts[0].lower()):
        if isinstance(city, str) and city.strip():
            if city in city_to_district:
                return city_to_district[city]
            return city
        return None
    # if parts are like 'Town, District, Province' -> pick middle as district
    if len(parts) >= 3:
        return parts[-2]
    # if len==2 and first is likely a district/city, return first
    if len(parts) == 2:
        first = parts[0]
        if city and isinstance(city, str) and city.lower() == first.lower():
            return city_to_district.get(city) or first
        return first
    # fallback: if the single token is not a province, return it
    if len(parts) == 1:
        single = parts[0]
        if any(p.lower() == single.lower() for p in provinces) or 'province' in single.lower():
            return None
        return single
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
