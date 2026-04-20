"""
build_features.py
-----------------
Reads a raw dataset with columns [url, label/status] and extracts all 30
features, producing a feature CSV ready for model training.

Usage:
    python utils/build_features.py
    python utils/build_features.py --input data/phishing_dataset_v1.csv \
                                   --output data/phishing_dataset_v1_features.csv
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feature_extraction import extract_features, FEATURE_COLUMNS

DEFAULT_INPUT  = ROOT / "data" / "phishing_dataset_v1.csv"
DEFAULT_OUTPUT = ROOT / "data" / "phishing_dataset_v1_features.csv"

# Accepted label column names (mapped to 'label')
LABEL_ALIASES = ["label", "status", "class", "target", "phishing"]


def _detect_label_col(columns: list) -> str:
    for alias in LABEL_ALIASES:
        if alias in columns:
            return alias
    raise ValueError(
        f"Could not find a label column. Expected one of {LABEL_ALIASES}. "
        f"Found: {columns}"
    )


def build_features(
    input_path: str = str(DEFAULT_INPUT),
    output_path: str = str(DEFAULT_OUTPUT),
    verbose: bool = True,
) -> pd.DataFrame:
    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {input_path}")

    df_raw = pd.read_csv(input_path)

    # Detect url column
    if "url" not in df_raw.columns:
        raise ValueError(f"Input CSV must have a 'url' column. Found: {df_raw.columns.tolist()}")

    # Detect label column (handles 'label', 'status', etc.)
    label_col = _detect_label_col(df_raw.columns.tolist())
    if verbose and label_col != "label":
        print(f"  ℹ️  Label column detected as '{label_col}' → mapped to 'label'")

    df_raw = df_raw[["url", label_col]].dropna().reset_index(drop=True)
    df_raw = df_raw.rename(columns={label_col: "label"})

    if verbose:
        print(f"\n📂 Raw dataset   : {input_path.name}")
        print(f"   Rows          : {len(df_raw):,}")
        lv = df_raw["label"].value_counts()
        print(f"   Legitimate (0): {lv.get(0, 0):,}")
        print(f"   Phishing   (1): {lv.get(1, 0):,}")
        print(f"\n⚙️  Extracting 30 features ...")

    records = []
    urls   = df_raw["url"].tolist()
    labels = df_raw["label"].tolist()
    total  = len(urls)
    errors = 0

    for i, (url, label) in enumerate(zip(urls, labels)):
        try:
            feats = extract_features(str(url))
            feats["label"] = int(label)
            records.append(feats)
        except Exception as e:
            errors += 1
            if verbose:
                print(f"  ⚠️  Skipped '{url[:60]}': {e}")
        if verbose and (i + 1) % 10000 == 0:
            print(f"  ... {i+1:,}/{total:,} URLs processed")

    df = pd.DataFrame(records)
    df = df[FEATURE_COLUMNS + ["label"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"\n✅ Feature dataset saved → {output_path}")
        print(f"   Rows    : {len(df):,}")
        print(f"   Columns : {len(df.columns)} (30 features + label)")
        if errors:
            print(f"   Skipped : {errors} URLs (parse errors)")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from raw URL dataset")
    parser.add_argument("--input",  default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()
    build_features(args.input, args.output, verbose=not args.quiet)
