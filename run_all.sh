#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# run_all.sh  —  Full pipeline launcher for Phishing Detector
# ─────────────────────────────────────────────────────────────
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   🛡️  Phishing Detector — Setup & Run    ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# 1. Install dependencies
echo "▶ Installing dependencies …"
pip install -r requirements.txt -q
echo "  ✓ Dependencies installed"
echo ""

# 2. Generate dataset (if needed)
if [ ! -f "data/phishing_dataset.csv" ]; then
  echo "▶ Generating synthetic dataset …"
  python3 utils/generate_dataset.py
  echo ""
fi

# 3. Train models
echo "▶ Training models …"
python3 src/train_model.py
echo ""

# 4. Evaluate
echo "▶ Running evaluation …"
python3 src/evaluate.py
echo ""

# 5. Choose interface
echo "╔══════════════════════════════════════════╗"
echo "║  What would you like to launch?          ║"
echo "║  [1] Streamlit web app  (port 8501)      ║"
echo "║  [2] FastAPI REST API   (port 8000)      ║"
echo "║  [3] Both                                ║"
echo "║  [4] Exit                                ║"
echo "╚══════════════════════════════════════════╝"
read -rp "Choice [1-4]: " choice

case "$choice" in
  1)
    echo "▶ Starting Streamlit …"
    streamlit run app/streamlit_app.py --server.port 8501
    ;;
  2)
    echo "▶ Starting FastAPI …"
    uvicorn app.api:app --reload --port 8000
    ;;
  3)
    echo "▶ Starting FastAPI in background …"
    uvicorn app.api:app --port 8000 &
    API_PID=$!
    echo "  API running (PID $API_PID) on http://localhost:8000"
    echo "▶ Starting Streamlit …"
    streamlit run app/streamlit_app.py --server.port 8501
    kill "$API_PID" 2>/dev/null || true
    ;;
  4)
    echo "Bye!"
    ;;
  *)
    echo "Invalid choice. Run individual commands manually."
    ;;
esac
