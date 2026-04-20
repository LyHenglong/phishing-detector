"""
FastAPI backend for phishing URL detection.

Run:
    uvicorn app.api:app --reload --port 8000

Endpoints:
    GET  /            → health check
    POST /predict     → single URL prediction
    POST /predict/batch → batch predictions
    GET  /model/info  → model metadata
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi            import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic           import BaseModel, HttpUrl, field_validator
from typing             import List, Optional
import json

from predict import PhishingPredictor

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Phishing Detector API",
    description="ML-powered phishing URL detection with 30 features.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
try:
    predictor = PhishingPredictor()
    MODEL_LOADED = True
except FileNotFoundError:
    predictor    = None
    MODEL_LOADED = False


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class URLRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def url_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("URL must not be empty")
        return v.strip()


class BatchURLRequest(BaseModel):
    urls: List[str]

    @field_validator("urls")
    @classmethod
    def urls_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("urls list must not be empty")
        if len(v) > 100:
            raise ValueError("Batch size limit is 100 URLs")
        return v


class PredictionResponse(BaseModel):
    url:           str
    label:         str
    is_phishing:   bool
    prob_phishing: float
    confidence:    float
    risk_level:    str
    features:      dict


class ModelInfoResponse(BaseModel):
    model_name:   str
    test_metrics: dict
    feature_cols: List[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def root():
    return {
        "service": "Phishing Detector API",
        "status":  "ok" if MODEL_LOADED else "model_not_loaded",
        "version": "1.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["health"])
def health():
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_url(req: URLRequest):
    """Predict whether a single URL is phishing."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        result = predictor.predict(req.url)
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/batch", tags=["prediction"])
def predict_batch(req: BatchURLRequest):
    """Predict phishing for a list of URLs (max 100)."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        results = predictor.predict_batch(req.urls)
        return {"count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info():
    """Return model name, test metrics, and feature list."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_name":   predictor.model_name,
        "test_metrics": predictor.meta.get("test_metrics", {}),
        "feature_cols": predictor.feature_cols,
    }


@app.get("/model/features", tags=["model"])
def feature_descriptions():
    """Return the 30 feature names with descriptions."""
    descriptions = {
        "url_length": "Total character count of the URL",
        "domain_length": "Length of the registered domain",
        "path_length": "Length of the URL path component",
        "query_length": "Length of the query string",
        "num_dots": "Count of '.' characters",
        "num_hyphens": "Count of '-' characters",
        "num_underscores": "Count of '_' characters",
        "num_slashes": "Count of '/' characters",
        "num_at_symbols": "Count of '@' (can redirect to attacker host)",
        "num_ampersands": "Count of '&' in query params",
        "num_equals_signs": "Count of '=' in query params",
        "num_question_marks": "Count of '?'",
        "num_special_chars": "Count of non-standard special characters",
        "num_digits": "Absolute digit count",
        "digit_ratio": "Fraction of digits in the URL",
        "has_https": "1 if scheme is HTTPS",
        "has_ip_address": "1 if IPv4 address appears in URL",
        "num_subdomains": "Number of subdomain levels",
        "domain_entropy": "Shannon entropy of the domain string",
        "has_suspicious_keyword": "1 if any phishing keyword found",
        "num_suspicious_keywords": "Count of phishing-related keywords",
        "has_suspicious_tld": "1 if TLD is from known-abused list",
        "has_phishing_extension": "1 if path ends in .php, .asp, etc.",
        "path_depth": "Number of path segments",
        "has_port": "1 if a non-standard port is specified",
        "url_entropy": "Shannon entropy of the full URL",
        "token_count": "Number of alphanumeric tokens",
        "longest_token_length": "Length of the longest alphanumeric token",
        "has_double_slash_redirect": "1 if '//' appears in path",
        "has_hex_encoding": "1 if URL contains %XX percent-encoding",
    }
    return descriptions
