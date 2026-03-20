import argparse
import os

import pandas as pd


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compare_runs(file_a: str, file_b: str, label_a: str, label_b: str, mismatches_out: str = "") -> None:
    df_a = pd.read_csv(file_a, encoding="latin1")
    df_b = pd.read_csv(file_b, encoding="latin1")

    required_cols = {
        "User_Location",
        "user_country",
        "user_other",
        "user_location_parse_method",
        "user_location_parse_confidence",
    }

    for name, df in [(label_a, df_a), (label_b, df_b)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} file is missing required columns: {sorted(missing)}")

    if len(df_a) != len(df_b):
        raise ValueError("Input files have different row counts. Ensure both runs used same source CSV.")

    country_a = df_a["user_country"].fillna("").astype(str).str.strip()
    country_b = df_b["user_country"].fillna("").astype(str).str.strip()
    other_a = df_a["user_other"].fillna("").astype(str).str.strip()
    other_b = df_b["user_other"].fillna("").astype(str).str.strip()

    resolved_a = country_a != ""
    resolved_b = country_b != ""

    both_resolved = resolved_a & resolved_b
    either_resolved = resolved_a | resolved_b
    unresolved_both = (~resolved_a) & (~resolved_b)

    country_match_all = country_a.str.lower() == country_b.str.lower()
    country_match_both_resolved = country_match_all & both_resolved
    other_match_both_resolved = (other_a.str.lower() == other_b.str.lower()) & both_resolved

    llm_fill_a = (df_a["user_location_parse_method"].fillna("").astype(str) == "llm_fallback").sum()
    llm_fill_b = (df_b["user_location_parse_method"].fillna("").astype(str) == "llm_fallback").sum()

    print("=== Comparison Summary ===")
    print(f"Rows: {len(df_a)}")
    print(f"{label_a} resolved: {resolved_a.sum()} ({safe_ratio(int(resolved_a.sum()), len(df_a)):.2%})")
    print(f"{label_b} resolved: {resolved_b.sum()} ({safe_ratio(int(resolved_b.sum()), len(df_b)):.2%})")
    print(f"{label_a} LLM-filled rows: {int(llm_fill_a)}")
    print(f"{label_b} LLM-filled rows: {int(llm_fill_b)}")
    print(f"Both resolved: {int(both_resolved.sum())}")
    print(f"Either resolved: {int(either_resolved.sum())}")
    print(f"Both unresolved: {int(unresolved_both.sum())}")
    print(
        f"Country agreement (both resolved): {int(country_match_both_resolved.sum())}/"
        f"{int(both_resolved.sum())} ({safe_ratio(int(country_match_both_resolved.sum()), int(both_resolved.sum())):.2%})"
    )
    print(
        f"Other agreement (both resolved): {int(other_match_both_resolved.sum())}/"
        f"{int(both_resolved.sum())} ({safe_ratio(int(other_match_both_resolved.sum()), int(both_resolved.sum())):.2%})"
    )

    mismatch_mask = (
        (country_a.str.lower() != country_b.str.lower())
        | (other_a.str.lower() != other_b.str.lower())
    )
    mismatches = pd.DataFrame(
        {
            "User_Location": df_a["User_Location"],
            f"{label_a}_country": country_a.replace("", pd.NA),
            f"{label_a}_other": other_a.replace("", pd.NA),
            f"{label_a}_method": df_a["user_location_parse_method"],
            f"{label_a}_confidence": df_a["user_location_parse_confidence"],
            f"{label_b}_country": country_b.replace("", pd.NA),
            f"{label_b}_other": other_b.replace("", pd.NA),
            f"{label_b}_method": df_b["user_location_parse_method"],
            f"{label_b}_confidence": df_b["user_location_parse_confidence"],
        }
    )[mismatch_mask]

    print(f"Mismatched rows: {len(mismatches)}")

    if mismatches_out:
        out_dir = os.path.dirname(os.path.abspath(mismatches_out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        mismatches.to_csv(mismatches_out, index=False)
        print(f"Wrote mismatches: {mismatches_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two user-location split output CSVs from different LLM runs.",
    )
    parser.add_argument("--file-a", required=True, help="Path to first output CSV")
    parser.add_argument("--file-b", required=True, help="Path to second output CSV")
    parser.add_argument("--label-a", default="run_a", help="Label for first file")
    parser.add_argument("--label-b", default="run_b", help="Label for second file")
    parser.add_argument(
        "--mismatches-out",
        default="",
        help="Optional path to save row-level mismatches as CSV",
    )

    args = parser.parse_args()
    compare_runs(
        file_a=args.file_a,
        file_b=args.file_b,
        label_a=args.label_a,
        label_b=args.label_b,
        mismatches_out=args.mismatches_out,
    )


if __name__ == "__main__":
    main()
