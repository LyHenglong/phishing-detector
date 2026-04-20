"""
Synthetic phishing dataset generator.
Produces a balanced CSV with 30 features + label.
Replace with real data (e.g. PhishTank + Alexa top-1M) for production use.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

FEATURE_COLUMNS = [
    "url_length", "domain_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_underscores", "num_slashes",
    "num_at_symbols", "num_ampersands", "num_equals_signs", "num_question_marks",
    "num_special_chars", "num_digits", "digit_ratio",
    "has_https", "has_ip_address", "num_subdomains", "domain_entropy",
    "has_suspicious_keyword", "num_suspicious_keywords",
    "has_suspicious_tld", "has_phishing_extension",
    "path_depth", "has_port", "url_entropy",
    "token_count", "longest_token_length",
    "has_double_slash_redirect", "has_hex_encoding",
]


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate_legitimate(n: int) -> pd.DataFrame:
    records = {}
    records["url_length"]           = _clip(rng.normal(60, 20, n).astype(int), 10, 150)
    records["domain_length"]        = _clip(rng.normal(12, 5, n).astype(int), 3, 40)
    records["path_length"]          = _clip(rng.normal(18, 12, n).astype(int), 0, 80)
    records["query_length"]         = _clip(rng.exponential(10, n).astype(int), 0, 60)
    records["num_dots"]             = _clip(rng.poisson(2.5, n), 1, 8)
    records["num_hyphens"]          = _clip(rng.poisson(0.5, n), 0, 4)
    records["num_underscores"]      = _clip(rng.poisson(0.3, n), 0, 3)
    records["num_slashes"]          = _clip(rng.poisson(3, n), 1, 10)
    records["num_at_symbols"]       = rng.choice([0, 1], n, p=[0.98, 0.02])
    records["num_ampersands"]       = _clip(rng.poisson(1, n), 0, 6)
    records["num_equals_signs"]     = _clip(rng.poisson(1, n), 0, 6)
    records["num_question_marks"]   = rng.choice([0, 1], n, p=[0.6, 0.4])
    records["num_special_chars"]    = _clip(rng.poisson(1, n), 0, 6)
    records["num_digits"]           = _clip(rng.poisson(3, n), 0, 15)
    records["digit_ratio"]          = _clip(rng.beta(1, 8, n), 0, 0.4)
    records["has_https"]            = rng.choice([0, 1], n, p=[0.1, 0.9])
    records["has_ip_address"]       = rng.choice([0, 1], n, p=[0.99, 0.01])
    records["num_subdomains"]       = _clip(rng.poisson(1, n), 0, 3)
    records["domain_entropy"]       = _clip(rng.normal(3.2, 0.5, n), 1.5, 4.5)
    records["has_suspicious_keyword"]  = rng.choice([0, 1], n, p=[0.95, 0.05])
    records["num_suspicious_keywords"] = records["has_suspicious_keyword"] * rng.choice([1, 2], n, p=[0.8, 0.2])
    records["has_suspicious_tld"]   = rng.choice([0, 1], n, p=[0.98, 0.02])
    records["has_phishing_extension"] = rng.choice([0, 1], n, p=[0.92, 0.08])
    records["path_depth"]           = _clip(rng.poisson(2, n), 0, 7)
    records["has_port"]             = rng.choice([0, 1], n, p=[0.97, 0.03])
    records["url_entropy"]          = _clip(rng.normal(3.8, 0.5, n), 2.0, 5.5)
    records["token_count"]          = _clip(rng.poisson(6, n), 1, 20)
    records["longest_token_length"] = _clip(rng.normal(8, 3, n).astype(int), 2, 25)
    records["has_double_slash_redirect"] = rng.choice([0, 1], n, p=[0.97, 0.03])
    records["has_hex_encoding"]     = rng.choice([0, 1], n, p=[0.96, 0.04])
    records["label"]                = np.zeros(n, dtype=int)
    return pd.DataFrame(records)


def generate_phishing(n: int) -> pd.DataFrame:
    records = {}
    records["url_length"]           = _clip(rng.normal(110, 35, n).astype(int), 30, 300)
    records["domain_length"]        = _clip(rng.normal(22, 10, n).astype(int), 5, 60)
    records["path_length"]          = _clip(rng.normal(45, 20, n).astype(int), 0, 150)
    records["query_length"]         = _clip(rng.exponential(25, n).astype(int), 0, 120)
    records["num_dots"]             = _clip(rng.poisson(4.5, n), 1, 15)
    records["num_hyphens"]          = _clip(rng.poisson(2.5, n), 0, 12)
    records["num_underscores"]      = _clip(rng.poisson(1.5, n), 0, 8)
    records["num_slashes"]          = _clip(rng.poisson(6, n), 1, 20)
    records["num_at_symbols"]       = rng.choice([0, 1], n, p=[0.7, 0.3])
    records["num_ampersands"]       = _clip(rng.poisson(3, n), 0, 15)
    records["num_equals_signs"]     = _clip(rng.poisson(3, n), 0, 15)
    records["num_question_marks"]   = rng.choice([0, 1, 2], n, p=[0.3, 0.5, 0.2])
    records["num_special_chars"]    = _clip(rng.poisson(5, n), 0, 20)
    records["num_digits"]           = _clip(rng.poisson(9, n), 0, 40)
    records["digit_ratio"]          = _clip(rng.beta(3, 5, n), 0.05, 0.7)
    records["has_https"]            = rng.choice([0, 1], n, p=[0.55, 0.45])
    records["has_ip_address"]       = rng.choice([0, 1], n, p=[0.65, 0.35])
    records["num_subdomains"]       = _clip(rng.poisson(3, n), 0, 8)
    records["domain_entropy"]       = _clip(rng.normal(4.1, 0.6, n), 2.0, 5.5)
    records["has_suspicious_keyword"]  = rng.choice([0, 1], n, p=[0.2, 0.8])
    records["num_suspicious_keywords"] = records["has_suspicious_keyword"] * _clip(rng.poisson(2, n), 1, 6)
    records["has_suspicious_tld"]   = rng.choice([0, 1], n, p=[0.45, 0.55])
    records["has_phishing_extension"] = rng.choice([0, 1], n, p=[0.45, 0.55])
    records["path_depth"]           = _clip(rng.poisson(5, n), 0, 15)
    records["has_port"]             = rng.choice([0, 1], n, p=[0.75, 0.25])
    records["url_entropy"]          = _clip(rng.normal(4.5, 0.6, n), 2.5, 6.0)
    records["token_count"]          = _clip(rng.poisson(12, n), 2, 40)
    records["longest_token_length"] = _clip(rng.normal(18, 6, n).astype(int), 4, 50)
    records["has_double_slash_redirect"] = rng.choice([0, 1], n, p=[0.55, 0.45])
    records["has_hex_encoding"]     = rng.choice([0, 1], n, p=[0.45, 0.55])
    records["label"]                = np.ones(n, dtype=int)
    return pd.DataFrame(records)


def generate_dataset(n_samples: int = 20_000, output_path: str = None) -> pd.DataFrame:
    half = n_samples // 2
    legit   = generate_legitimate(half)
    phish   = generate_phishing(half)
    df = pd.concat([legit, phish], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Enforce column order
    df = df[FEATURE_COLUMNS + ["label"]]

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Dataset saved → {output_path}  ({len(df):,} rows)")

    return df


if __name__ == "__main__":
    generate_dataset(
        n_samples=20_000,
        output_path=str(Path(__file__).resolve().parents[1] / "data" / "phishing_dataset.csv"),
    )
