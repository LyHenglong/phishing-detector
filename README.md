<div align="center">

# 🛡️ Phishing URL Detector

**A machine learning pipeline for detecting phishing URLs** from 30
hand-crafted structural features extracted directly from the URL string —
no page fetching, no third-party lookups.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📌 Overview

This project trains a phishing URL classifier on **109,614 real labeled URLs**
(balanced: 54,807 phishing / 54,807 legitimate). The pipeline covers raw data
ingestion, automated feature extraction, multi-model training with a
leakage-safe evaluation split, and a REST API plus an interactive web
interface for inference.

> **Best model:** Gradient Boosting — **F1 0.9398 · ROC-AUC 0.9827**,
> measured on domains never seen during training (see
> [Evaluation methodology](#-evaluation-methodology) below).

---

## 📁 Project Structure

```
phishing-detector/
├── data/
│   ├── phishing_dataset_v1.csv              # Raw dataset: [url, status]
│   └── phishing_dataset_v1_features.csv     # Auto-generated: 30 features + domain + label
├── src/
│   ├── feature_extraction.py                # 30-feature extractor
│   ├── data_split.py                        # Leakage-safe train/val/test split (grouped by domain)
│   ├── train_model.py                       # Full pipeline: extract → split → train → evaluate
│   ├── evaluate.py                          # Standalone evaluation report
│   └── predict.py                           # Real-time prediction (CLI + importable)
├── app/
│   ├── api.py                               # FastAPI REST backend
│   └── streamlit_app.py                     # Streamlit web interface
├── models/
│   ├── best_model.pkl                       # Saved best model
│   ├── scaler.pkl                           # Fitted StandardScaler
│   ├── metadata.json                        # Model name, metrics, feature list
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── feature_importance.png
├── utils/
│   ├── build_features.py                    # Extracts 30 features (+ domain) from raw CSV
│   └── generate_dataset.py                  # Synthetic data generator (fallback/demo only)
├── notebooks/
│   └── exploration.ipynb                    # EDA, feature correlations, sample predictions
├── requirements.txt
├── run.sh                                   # Install → train (if needed) → launch
├── run_all.sh                               # Install → generate/train → evaluate → launch
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/LyHenglong/phishing-detector.git
cd phishing-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`xgboost` and `lightgbm` are optional — if installed, `train_model.py`
includes them as extra candidate models; if not, it falls back to Logistic
Regression, Random Forest, and Gradient Boosting only.

### 3. Train the model

The raw dataset uses columns `url` and `status` (0 = legitimate,
1 = phishing). The label column name is auto-detected — no manual config
needed.

```bash
python src/train_model.py
```

This automatically:
1. Extracts 30 features + the row's domain from every raw URL → saves `data/phishing_dataset_v1_features.csv`
2. Splits into train / val / test (70 / 15 / 15), **grouped by domain** so no domain appears in more than one split
3. Trains the candidate classifiers and picks the best by F1 on the validation set
4. Saves the model, scaler, metadata, and evaluation plots to `models/`

> To run feature extraction independently:
> ```bash
> python utils/build_features.py \
>   --input  data/phishing_dataset_v1.csv \
>   --output data/phishing_dataset_v1_features.csv
> ```

### 4. Evaluate the saved model

```bash
python src/evaluate.py
```

Reproduces the exact same held-out test split as `train_model.py` and
regenerates `models/evaluation_report.png` with precision-recall and ROC
curves.

### 5. Launch the web app

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501)

### 6. Launch the REST API

```bash
uvicorn app.api:app --reload --port 8000
```

- Swagger docs → [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc → [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8632 | 0.8465 | 0.8266 | 0.8364 | 0.9278 |
| Random Forest | 0.9282 | 0.9127 | 0.9181 | 0.9154 | 0.9762 |
| **Gradient Boosting** ✅ | **0.9500** | **0.9587** | **0.9217** | **0.9398** | **0.9827** |

All numbers are on the held-out test split (14,630 rows), on domains the
model never saw during training.

### 🔎 Evaluation methodology

Many rows in the raw dataset share a domain — the same abused site
(free-hosting / IPFS gateways, Google Docs links, etc.) hosts several
phishing paths, and a handful of large legitimate sites (Wikipedia, YouTube,
Facebook, ...) also appear many times. Splitting train/val/test **by row**
lets the same domain leak into both sides: the model can partly memorize a
domain's identity via domain-level features instead of learning transferable
URL patterns. That produced an earlier, inflated result of **F1 0.9638 / AUC
0.9934** — nearly identical across train, validation, and test, which is
itself a tell that the split wasn't actually separating "seen" from
"unseen" data.

`src/data_split.py` now splits **at the domain level**: unique domains are
partitioned into train/val/test stratified by each domain's majority label
(domains are ~99% label-pure), so no domain appears in more than one split.
The table above reflects genuine generalization to domains not seen during
training — a more honest, if more modest, number.

---

## 🔍 Features (30 total)

All features are extracted from the raw URL string — no HTTP requests are
made.

| # | Category | Features |
|---|---|---|
| 1–11 | **URL structure** | `url_length`, `domain_length`, `path_length`, `query_length`, `num_dots`, `num_slashes`, `has_https`, `has_ip_address`, `num_subdomains`, `path_depth`, `has_port` |
| 12–20 | **Character patterns** | `num_hyphens`, `num_underscores`, `num_at_symbols`, `num_ampersands`, `num_equals_signs`, `num_question_marks`, `num_special_chars`, `num_digits`, `digit_ratio` |
| 21–22 | **Domain signals** | `domain_entropy`, `has_suspicious_tld` |
| 23–25 | **Content signals** | `has_suspicious_keyword`, `num_suspicious_keywords`, `has_phishing_extension` |
| 26–27 | **Obfuscation** | `has_hex_encoding`, `has_double_slash_redirect` |
| 28–30 | **Entropy & tokens** | `url_entropy`, `token_count`, `longest_token_length` |

The feature CSV also carries a `domain` column used only for the train/val/test
split above — it is not fed to the model as a feature.

---

## 🔌 API Reference

### `POST /predict` — Single URL

**Request:**
```json
{ "url": "http://paypal-verify.xyz/login.php?user=test" }
```

**Response:**
```json
{
  "url": "http://paypal-verify.xyz/login.php?user=test",
  "label": "phishing",
  "is_phishing": true,
  "prob_phishing": 0.9734,
  "confidence": 0.9734,
  "risk_level": "HIGH",
  "features": { "url_length": 47, "has_https": 0, "...": "..." }
}
```

Risk levels: `SAFE` · `LOW` · `MEDIUM` · `HIGH`

---

### `POST /predict/batch` — Multiple URLs (max 100)

**Request:**
```json
{ "urls": ["https://google.com", "http://phish.tk/login.php"] }
```

**Response:** `{ "count": 2, "results": [ { ...same shape as /predict... }, ... ] }`

---

### `GET /model/info` — Model name, test metrics, and feature list

### `GET /model/features` — The 30 feature names with one-line descriptions

### `GET /health` — Returns 200 once a trained model is loaded, 503 otherwise

---

## 💻 Python Usage

```python
from src.predict import PhishingPredictor

predictor = PhishingPredictor()

# Single prediction
result = predictor.predict("http://paypal-verify.xyz/login.php?user=test")
print(result["risk_level"])      # HIGH
print(result["prob_phishing"])   # 0.97

# Batch prediction
results = predictor.predict_batch([
    "https://www.google.com",
    "http://secure-ebay-signin.tk/signin/?redirect=%2F",
])
```

---

## 🖥️ CLI Usage

```bash
# Single URL
python src/predict.py "http://paypal-verify.xyz/login.php"

# Batch from file (one URL per line)
python src/predict.py --batch urls.txt

# JSON output
python src/predict.py "http://phish.xyz" --json
```

---

## 🔄 Bring Your Own Dataset

The pipeline accepts any CSV with a URL column and a label column. Supported
label column names are auto-detected: `label`, `status`, `class`, `target`,
`phishing`.

```bash
python utils/build_features.py \
  --input  data/your_dataset.csv \
  --output data/your_dataset_features.csv

python src/train_model.py \
  --raw   data/your_dataset.csv \
  --feats data/your_dataset_features.csv
```

If your dataset also has many rows per domain, the domain-grouped split in
`src/data_split.py` applies automatically — no extra setup needed.

Recommended real-world data sources:
- [PhishTank](https://www.phishtank.com/developer_info.php) — free phishing URL feed
- [OpenPhish](https://openphish.com) — phishing intelligence feed
- [Alexa / Majestic Top 1M](http://s3.amazonaws.com/alexa-static/top-1m.csv.zip) — legitimate URLs

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML models | Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting); XGBoost, LightGBM *(optional)* |
| Feature engineering | Pure Python (`urllib`, `re`, `math`) |
| REST API | FastAPI + Uvicorn |
| Web interface | Streamlit |
| Serialization | Joblib |
| Visualization | Matplotlib, Seaborn |

---

## ⚠️ Known limitations

- **URL structure only.** This tool analyzes the URL string — it does not
  fetch, render, or visit any web page, and has no access to page content,
  SSL certificate details, WHOIS data, or redirect chains. It is intended for
  research and educational purposes and should not be the sole basis for
  security decisions.
- **Dataset source skew.** Legitimate and phishing URLs in the training data
  come from different sources, which can introduce patterns the model learns
  that don't reflect "phishing-ness" per se. The domain-grouped split (above)
  catches same-domain leakage but does not eliminate this broader
  distribution-shift risk — performance on a fresh, independently collected
  feed (e.g. a live PhishTank pull) should be expected to differ from the
  numbers reported here.
- **Static snapshot.** Phishing campaigns evolve; a model trained on a fixed
  dataset will drift. Retrain periodically on fresh data for production use.

---

## 📄 License

MIT — see [LICENSE](LICENSE). © 2026 Henglong LY
