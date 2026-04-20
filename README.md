<div align="center">

# 🛡️ Phishing URL Detector — v1

**A production-grade machine learning pipeline for detecting phishing URLs**
using 30 hand-crafted structural features extracted directly from raw URLs.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📌 Overview

This project trains a phishing URL classifier on **109,614 real labeled URLs** (balanced: 54,807 phishing / 54,807 legitimate). The pipeline covers everything from raw data ingestion, automated feature extraction, multi-model training, and evaluation, to a REST API and an interactive web interface.

> **Best model:** Gradient Boosting — **F1: 0.9638 · AUC: 0.9934**

---

## 📁 Project Structure

```
phishing_detector_v1/
├── data/
│   ├── phishing_dataset_v1.csv              # Raw dataset: [url, status]
│   └── phishing_dataset_v1_features.csv     # Auto-generated: 30 features + label
├── src/
│   ├── feature_extraction.py                # 30-feature extractor
│   ├── train_model.py                       # Full pipeline: extract → train → evaluate
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
│   ├── build_features.py                    # Extracts 30 features from raw CSV
│   └── generate_dataset.py                  # Synthetic data generator (fallback)
├── notebooks/
│   └── exploration.ipynb                    # EDA, distributions, correlation matrix
├── requirements.txt
├── run.sh
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/phishing_detector_v1.git
cd phishing_detector_v1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

The raw dataset uses columns `url` and `status` (0 = legitimate, 1 = phishing). The label column name is auto-detected — no manual config needed.

```bash
python src/train_model.py
```

This automatically:
1. Extracts 30 features from every raw URL → saves `data/phishing_dataset_v1_features.csv`
2. Splits data into train / val / test (70 / 15 / 15)
3. Trains 3–5 classifiers and picks the best by F1 on the validation set
4. Saves the model, scaler, and evaluation plots to `models/`

> To run feature extraction independently:
> ```bash
> python utils/build_features.py \
>   --input  data/phishing_dataset_v1.csv \
>   --output data/phishing_dataset_v1_features.csv
> ```

### 4. Launch the web app

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501)

### 5. Launch the REST API

```bash
uvicorn app.api:app --reload --port 8000
```

- Swagger docs → [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc → [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📊 Results

Trained on **109,614 real URLs** — 10× larger than v0, with a significant accuracy boost:

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8568 | 0.8685 | 0.8410 | 0.8545 | 0.9335 |
| Random Forest | 0.9624 | 0.9768 | 0.9473 | 0.9618 | 0.9931 |
| **Gradient Boosting** ✅ | **0.9643** | **0.9764** | **0.9516** | **0.9638** | **0.9934** |

**v1 vs v0 comparison:**

| Metric | v0 (11k URLs) | v1 (109k URLs) | Δ |
|---|---|---|---|
| F1 | 0.8878 | 0.9638 | +0.0760 |
| AUC | 0.9540 | 0.9934 | +0.0394 |
| Accuracy | 0.8892 | 0.9643 | +0.0751 |

---

## 🔍 Features (30 total)

All features are extracted from the raw URL string — no HTTP requests are made.

| # | Category | Features |
|---|---|---|
| 1–11 | **URL structure** | `url_length`, `domain_length`, `path_length`, `query_length`, `num_dots`, `num_slashes`, `has_https`, `has_ip_address`, `num_subdomains`, `path_depth`, `has_port` |
| 12–20 | **Character patterns** | `num_hyphens`, `num_underscores`, `num_at_symbols`, `num_ampersands`, `num_equals_signs`, `num_question_marks`, `num_special_chars`, `num_digits`, `digit_ratio` |
| 21–22 | **Domain signals** | `domain_entropy`, `has_suspicious_tld` |
| 23–25 | **Content signals** | `has_suspicious_keyword`, `num_suspicious_keywords`, `has_phishing_extension` |
| 26–27 | **Obfuscation** | `has_hex_encoding`, `has_double_slash_redirect` |
| 28–30 | **Entropy & tokens** | `url_entropy`, `token_count`, `longest_token_length` |

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
  "label": "phishing",
  "is_phishing": true,
  "prob_phishing": 0.9734,
  "risk_level": "HIGH",
  "confidence": 0.9734,
  "features": { "url_length": 47, "has_https": 0, "..." : "..." }
}
```

Risk levels: `SAFE` · `LOW` · `MEDIUM` · `HIGH`

---

### `POST /predict/batch` — Multiple URLs

**Request:**
```json
{ "urls": ["https://google.com", "http://phish.tk/login.php"] }
```

---

### `GET /model/info` — Model metadata and test metrics

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

The pipeline accepts any CSV with a URL column and a label column. Supported label column names are auto-detected: `label`, `status`, `class`, `target`, `phishing`.

```bash
python utils/build_features.py \
  --input  data/your_dataset.csv \
  --output data/your_dataset_features.csv

python src/train_model.py \
  --raw   data/your_dataset.csv \
  --feats data/your_dataset_features.csv
```

Recommended real-world data sources:
- [PhishTank](https://www.phishtank.com/developer_info.php) — free phishing URL feed
- [OpenPhish](https://openphish.com) — phishing intelligence feed
- [Alexa / Majestic Top 1M](http://s3.amazonaws.com/alexa-static/top-1m.csv.zip) — legitimate URLs

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML models | Scikit-learn, XGBoost, LightGBM |
| Feature engineering | Pure Python (`urllib`, `re`, `math`) |
| REST API | FastAPI + Uvicorn |
| Web interface | Streamlit |
| Serialization | Joblib |
| Visualization | Matplotlib, Seaborn |

---

## ⚠️ Disclaimer

This tool analyzes URL structure only — it does not fetch, render, or visit any web page. It is intended for research and educational purposes and should not be used as the sole basis for security decisions.

---

## 📄 License

©️Copyright2026 || Henglong LY

