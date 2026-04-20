"""
evaluate.py
-----------
Standalone evaluation script.
Loads the saved best model + scaler, runs evaluation on the test split,
and prints / re-saves a report.

Usage:
    python src/evaluate.py
    python src/evaluate.py --data data/phishing_dataset.csv --models models/
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
)

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data"   / "phishing_dataset.csv"
MODELS = ROOT / "models"


def evaluate(data_path: Path = DATA, model_dir: Path = MODELS):
    # ── Load artefacts ────────────────────────────────────────────────────
    model_path  = model_dir / "best_model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    meta_path   = model_dir / "metadata.json"

    for p in (model_path, scaler_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}  — run train_model.py first.")

    clf    = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    model_name   = meta["best_model"]

    # ── Load & split (reproduce same split) ──────────────────────────────
    df = pd.read_csv(data_path)
    X  = df[feature_cols].values
    y  = df["label"].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_test = scaler.transform(X_test)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # ── Console report ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Evaluation Report — {model_name}")
    print(f"{'='*60}")
    print(f"\n  Test samples : {len(y_test):,}")
    print(f"  Phishing     : {y_test.sum():,}")
    print(f"  Legitimate   : {(y_test == 0).sum():,}\n")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print(f"  ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  Avg-P   : {average_precision_score(y_test, y_prob):.4f}")

    # ── Precision-Recall curve ────────────────────────────────────────────
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(rec, prec, color="#2563eb")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].grid(alpha=0.3)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color="#16a34a",
                 label=f"AUC={roc_auc_score(y_test, y_prob):.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"Evaluation — {model_name}", y=1.02)
    plt.tight_layout()
    out = model_dir / "evaluation_report.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved evaluation report → {out}")

    # ── Update metadata with eval results ─────────────────────────────────
    meta["evaluation"] = {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return y_test, y_pred, y_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved phishing model")
    parser.add_argument("--data",   default=str(DATA),   help="Dataset CSV path")
    parser.add_argument("--models", default=str(MODELS), help="Model directory")
    args = parser.parse_args()
    evaluate(Path(args.data), Path(args.models))
