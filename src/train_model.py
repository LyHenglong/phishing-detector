"""
train_model.py  (v1)
--------------------
Full training pipeline:
  1. If feature CSV is missing, auto-run build_features.py on the raw dataset
  2. Split -> train / val / test
  3. Scale features
  4. Train multiple classifiers
  5. Evaluate on validation set -> pick best by F1
  6. Final evaluation on held-out test set
  7. Save best model, scaler, metadata, and plots to models/
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
import joblib

try:
    from xgboost  import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

ROOT       = Path(__file__).resolve().parents[1]
RAW_DATA   = ROOT / "data" / "phishing_dataset_v1.csv"
FEAT_DATA  = ROOT / "data" / "phishing_dataset_v1_features.csv"
MODEL_DIR  = ROOT / "models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_feature_csv(raw_path: Path, feat_path: Path):
    if feat_path.exists():
        print(f"  Feature CSV found: {feat_path.name}")
        return
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Neither feature CSV ({feat_path}) nor raw dataset ({raw_path}) found.\n"
            f"Place your raw [url, status] CSV at: {raw_path}"
        )
    print(f"  Feature CSV not found — running feature extraction on {raw_path.name} ...")
    import sys
    sys.path.insert(0, str(ROOT / "utils"))
    from build_features import build_features
    build_features(str(raw_path), str(feat_path), verbose=True)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)),  4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred,    zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred,        zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)),  4),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix -> {path.name}")


def plot_roc_curves(results: dict, y_test, path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c"]
    for (name, info), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, info["prob"])
        ax.plot(fpr, tpr, label=f"{name}  (AUC={info['metrics']['roc_auc']:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved ROC curves -> {path.name}")


def plot_feature_importance(model, feature_names: list, path: Path):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        return
    idx   = np.argsort(imp)[-20:]
    names = [feature_names[i] for i in idx]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(names, imp[idx], color="#2563eb")
    ax.set_xlabel("Importance")
    ax.set_title("Top-20 Feature Importances")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved feature importance -> {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    raw_path:  Path = RAW_DATA,
    feat_path: Path = FEAT_DATA,
    model_dir: Path = MODEL_DIR,
):
    model_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Step 1: Feature CSV ──────────────────────────────────────")
    ensure_feature_csv(raw_path, feat_path)

    print("\n── Step 2: Load features ────────────────────────────────────")
    df = pd.read_csv(feat_path)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values
    print(f"  {len(df):,} rows  |  {len(feature_cols)} features  |  "
          f"phishing={y.sum():,}  legitimate={(y==0).sum():,}")

    print("\n── Step 3: Train / Val / Test split ─────────────────────────")
    X_tmp,  X_test,  y_tmp,  y_test  = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val,  y_train, y_val  = train_test_split(
        X_tmp, y_tmp, test_size=0.15/0.85, stratify=y_tmp, random_state=42)
    print(f"  Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")

    print("\n── Step 4: Scale ────────────────────────────────────────────")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, model_dir / "scaler.pkl")
    print("  Scaler fitted and saved.")

    print("\n── Step 5: Train models ─────────────────────────────────────")
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        "Random Forest":       RandomForestClassifier(
                                   n_estimators=300, max_depth=20,
                                   min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(
                                   n_estimators=200, max_depth=5,
                                   learning_rate=0.1, random_state=42),
    }
    if HAS_XGB:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
    if HAS_LGB:
        candidates["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            num_leaves=63, random_state=42, n_jobs=-1, verbose=-1,
        )

    val_results = {}
    for name, clf in candidates.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        elapsed = time.time() - t0
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]
        m = compute_metrics(y_val, y_pred, y_prob)
        val_results[name] = {"model": clf, "metrics": m, "time": elapsed}
        print(f"  {name:<22}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}  ({elapsed:.1f}s)")

    print("\n── Step 6: Pick best by F1 ──────────────────────────────────")
    best_name = max(val_results, key=lambda k: val_results[k]["metrics"]["f1"])
    best_clf  = val_results[best_name]["model"]
    print(f"  Best on validation: {best_name}")

    print("\n── Step 7: Test-set evaluation ──────────────────────────────")
    test_results = {}
    for name, info in val_results.items():
        clf    = info["model"]
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        m = compute_metrics(y_test, y_pred, y_prob)
        test_results[name] = {"metrics": m, "prob": y_prob}
        marker = " <- best" if name == best_name else ""
        print(f"  {name:<22}  Acc={m['accuracy']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}{marker}")

    print("\n── Step 8: Save artefacts ───────────────────────────────────")
    joblib.dump(best_clf, model_dir / "best_model.pkl")
    print(f"  Model saved -> best_model.pkl")

    metadata = {
        "version":         "v1",
        "best_model":      best_name,
        "feature_cols":    feature_cols,
        "raw_dataset":     str(raw_path.name),
        "feature_dataset": str(feat_path.name),
        "n_train":         int(len(X_train)),
        "n_val":           int(len(X_val)),
        "n_test":          int(len(X_test)),
        "test_metrics":    test_results[best_name]["metrics"],
        "all_models":      {n: v["metrics"] for n, v in test_results.items()},
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("  Metadata saved -> metadata.json")

    best_y_pred = best_clf.predict(X_test)
    plot_confusion_matrix(y_test, best_y_pred, best_name, model_dir / "confusion_matrix.png")
    plot_roc_curves(test_results, y_test, model_dir / "roc_curves.png")
    plot_feature_importance(best_clf, feature_cols, model_dir / "feature_importance.png")

    print(f"\n✓ Training complete  [v1]  Best model: {best_name}")
    print(f"  Test F1={test_results[best_name]['metrics']['f1']:.4f}  "
          f"AUC={test_results[best_name]['metrics']['roc_auc']:.4f}")
    return best_clf, scaler, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train phishing detection models (v1)")
    parser.add_argument("--raw",    default=str(RAW_DATA),   help="Raw [url,status] CSV")
    parser.add_argument("--feats",  default=str(FEAT_DATA),  help="Feature CSV (built if missing)")
    parser.add_argument("--models", default=str(MODEL_DIR),  help="Output directory for models")
    args = parser.parse_args()
    train(Path(args.raw), Path(args.feats), Path(args.models))
