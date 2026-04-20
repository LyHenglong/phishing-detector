#!/usr/bin/env bash
# run.sh — convenience script to set up and launch the phishing detector
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "🛡️  Phishing Detector — Setup & Launch"
echo "======================================="

# 1. Install deps
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

# 2. Train if model doesn't exist
if [ ! -f "models/best_model.pkl" ]; then
    echo ""
    echo "🤖 Training model (first run)..."
    python3 src/train_model.py
else
    echo ""
    echo "✅ Model already trained — skipping training."
fi

# 3. Launch choice
echo ""
echo "What would you like to launch?"
echo "  1) Streamlit web app   (recommended)"
echo "  2) FastAPI REST backend"
echo "  3) Both (requires two terminals)"
echo ""
read -rp "Choice [1/2/3]: " choice

case "$choice" in
  2)
    echo ""
    echo "🚀 Starting FastAPI on http://localhost:8000 ..."
    uvicorn app.api:app --reload --port 8000
    ;;
  3)
    echo ""
    echo "🚀 Starting FastAPI on http://localhost:8000 (background) ..."
    uvicorn app.api:app --port 8000 &
    API_PID=$!
    echo "🚀 Starting Streamlit on http://localhost:8501 ..."
    streamlit run app/streamlit_app.py
    kill $API_PID 2>/dev/null || true
    ;;
  *)
    echo ""
    echo "🚀 Starting Streamlit on http://localhost:8501 ..."
    streamlit run app/streamlit_app.py
    ;;
esac
