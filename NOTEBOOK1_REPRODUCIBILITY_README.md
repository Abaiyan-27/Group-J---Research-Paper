# Notebook 1 Reproducibility Worklog

Target notebook:
- Tourism Analysis 1 - Dataset Preparation.ipynb

## What was changed

1. Removed runtime dependency installation from notebook execution path.
- Deleted in-cell pip install logic.
- Notebook now assumes environment is prepared via requirements.txt.

2. Made country resolution offline-first by default.
- Added reproducibility flags in the country cell:
  - ENABLE_GEOCODING = False (default)
  - CACHE_PATH = country_resolution_cache.csv
  - SAVE_CACHE_UPDATES = True
- Resolution order is now:
  - direct country alias match
  - local cache lookup
  - optional geocoding (only if ENABLE_GEOCODING=True)
- Added resilient cache load/save helpers.

3. Added fail-fast schema validation before export.
- Final cleanup/export cell now validates all required output columns.
- Raises ValueError if required columns are missing.

4. Removed overwrite-prone output behavior.
- Final output now writes timestamped files:
  - Processed_Reviews_dataset_prep_YYYYMMDD_HHMMSS.csv
- This prevents accidental replacement of previous results.

5. Updated dependency declarations for reproducibility.
- requirements.txt switched from broad >= specs to pinned versions.
- Added explicit geopy pin:
  - geopy==2.4.1

6. Updated project documentation.
- Added "Notebook 1 Reproducibility Update" section in README.md.
- Included run guidance for deterministic offline runs.

## Validation performed

Notebook 1 cells were re-run in order after edits:
- Cell 2: data load and inspection -> success
- Cell 3: province/district extraction -> success
- Cell 4: country resolution (offline mode) -> success
- Cell 5: temporal features -> success
- Cell 6: rating class + review delay -> success
- Cell 7: schema check + export -> success

Observed export example:
- Processed_Reviews_dataset_prep_20260413_053030.csv

## New/updated artifacts generated during validation

- country_resolution_cache.csv (created)
- Processed_Reviews_dataset_prep_20260413_053030.csv (created)
- Processed_Reviews.csv (was already present; changed earlier during notebook execution history)

## How to run reproducibly

1. Install pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Open notebook and Run All from top:
- Tourism Analysis 1 - Dataset Preparation.ipynb

3. Keep default:
- ENABLE_GEOCODING = False

4. Use timestamped output file from the final cell for downstream steps.
