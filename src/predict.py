"""
predict.py
----------
Real-time URL phishing prediction.

As a module:
    from src.predict import PhishingPredictor
    p = PhishingPredictor()
    result = p.predict("http://suspicious-login.xyz/verify?account=123")

As a CLI:
    python src/predict.py http://suspicious-login.xyz/verify?account=123
    python src/predict.py --batch urls.txt
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(ROOT / "src"))
from feature_extraction import extract_features, extract_feature_vector, FEATURE_COLUMNS


class PhishingPredictor:
    """
    Loads the saved model + scaler and exposes .predict() / .predict_batch().
    Thread-safe; load once and reuse.
    """

    def __init__(self, model_dir: str = None):
        model_dir = Path(model_dir) if model_dir else ROOT / "models"
        self.clf    = joblib.load(model_dir / "best_model.pkl")
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        with open(model_dir / "metadata.json") as f:
            self.meta = json.load(f)
        self.feature_cols = self.meta["feature_cols"]
        self.model_name   = self.meta["best_model"]

    # ── Single prediction ─────────────────────────────────────────────────

    def predict(self, url: str) -> dict:
        """
        Predict whether a URL is phishing.

        Returns
        -------
        dict with keys:
          url, label, confidence, risk_level, features
        """
        features = extract_features(url)
        vec      = np.array([features[c] for c in self.feature_cols]).reshape(1, -1)
        vec_sc   = self.scaler.transform(vec)

        label      = int(self.clf.predict(vec_sc)[0])
        prob_phish = float(self.clf.predict_proba(vec_sc)[0, 1])
        confidence = prob_phish if label == 1 else 1 - prob_phish

        if prob_phish >= 0.80:
            risk = "HIGH"
        elif prob_phish >= 0.50:
            risk = "MEDIUM"
        elif prob_phish >= 0.25:
            risk = "LOW"
        else:
            risk = "SAFE"

        return {
            "url":         url,
            "label":       "phishing" if label == 1 else "legitimate",
            "is_phishing": label == 1,
            "prob_phishing": round(prob_phish, 4),
            "confidence":  round(confidence, 4),
            "risk_level":  risk,
            "features":    {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in features.items()},
        }

    # ── Batch prediction ──────────────────────────────────────────────────

    def predict_batch(self, urls: list[str]) -> list[dict]:
        return [self.predict(u) for u in urls]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _pretty(result: dict):
    icon = "🚨" if result["is_phishing"] else "✅"
    print(f"\n{icon}  {result['url']}")
    print(f"   Label      : {result['label'].upper()}")
    print(f"   Risk       : {result['risk_level']}")
    print(f"   P(phishing): {result['prob_phishing']:.1%}")
    print(f"   Confidence : {result['confidence']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Phishing URL predictor")
    parser.add_argument("url",        nargs="?",  help="Single URL to check")
    parser.add_argument("--batch",                help="Path to file with one URL per line")
    parser.add_argument("--models",   default=str(ROOT / "models"), help="Model directory")
    parser.add_argument("--json",     action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    predictor = PhishingPredictor(model_dir=args.models)

    if args.batch:
        urls = [l.strip() for l in open(args.batch) if l.strip()]
        results = predictor.predict_batch(urls)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                _pretty(r)
    elif args.url:
        result = predictor.predict(args.url)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _pretty(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
