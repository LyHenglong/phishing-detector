"""
Streamlit web interface for the Phishing Detector.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import json
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -1px;
    line-height: 1.2;
}
.subtitle { color: #475569; font-size: 1.05rem; margin-top: 4px; }

.risk-safe    { background:#dcfce7; color:#166534; border-left:4px solid #16a34a; }
.risk-low     { background:#fef9c3; color:#854d0e; border-left:4px solid #ca8a04; }
.risk-medium  { background:#ffedd5; color:#9a3412; border-left:4px solid #ea580c; }
.risk-high    { background:#fee2e2; color:#991b1b; border-left:4px solid #dc2626; }

.risk-card {
    border-radius: 10px;
    padding: 18px 22px;
    margin: 16px 0;
    font-size: 1.1rem;
    font-weight: 500;
}
.metric-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-value { font-size: 1.7rem; font-weight: 700; color: #1e293b; }
.metric-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: .5px; }

.feature-bar {
    height: 6px; border-radius: 3px;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    margin-top: 4px;
}
.stButton>button {
    background: #1e293b;
    color: white;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.88rem;
    letter-spacing: .3px;
    padding: 10px 28px;
    border: none;
    width: 100%;
    transition: background .2s;
}
.stButton>button:hover { background: #334155; }
</style>
""", unsafe_allow_html=True)


# ── Load predictor (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor():
    try:
        from predict import PhishingPredictor
        return PhishingPredictor(), None
    except FileNotFoundError as e:
        return None, str(e)


predictor, load_error = load_predictor()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Phishing Detector")
    st.caption("ML-powered URL analysis with 30 features")
    st.divider()

    if predictor:
        meta = predictor.meta
        st.markdown("**Model**")
        st.info(meta.get("best_model", "Unknown"))

        m = meta.get("test_metrics", {})
        if m:
            st.markdown("**Test Performance**")
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{m.get('accuracy',0):.1%}")
            col2.metric("F1 Score", f"{m.get('f1',0):.1%}")
            col1.metric("Precision", f"{m.get('precision',0):.1%}")
            col2.metric("Recall",    f"{m.get('recall',0):.1%}")
            st.metric("ROC-AUC", f"{m.get('roc_auc',0):.4f}")

        st.divider()

    mode = st.radio(
        "Mode",
        ["Single URL", "Batch Analysis", "Feature Explorer", "About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Built with Scikit-learn · XGBoost · Streamlit")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🛡️ Phishing URL Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze URLs in real-time using machine learning — 30 hand-crafted features.</p>', unsafe_allow_html=True)
st.divider()

if load_error:
    st.error(f"⚠️ Model not loaded: {load_error}")
    st.info("Run `python src/train_model.py` first, then restart Streamlit.")
    st.stop()


# ── Risk card helper ──────────────────────────────────────────────────────────
RISK_ICONS = {"SAFE": "✅", "LOW": "⚠️", "MEDIUM": "🟠", "HIGH": "🚨"}
RISK_CSS   = {"SAFE": "risk-safe", "LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}

def render_risk_card(result: dict):
    risk  = result["risk_level"]
    icon  = RISK_ICONS[risk]
    css   = RISK_CSS[risk]
    label = result["label"].upper()
    prob  = result["prob_phishing"]
    conf  = result["confidence"]
    st.markdown(
        f'<div class="risk-card {css}">'
        f'{icon}  <strong>{label}</strong> — Risk: <strong>{risk}</strong> &nbsp;|&nbsp; '
        f'P(phishing): <strong>{prob:.1%}</strong> &nbsp;|&nbsp; '
        f'Confidence: <strong>{conf:.1%}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_feature_table(features: dict, top_n: int = 15):
    """Show top-N features most indicative of phishing."""
    rows = []
    PHISHING_HIGH = {
        "has_suspicious_keyword", "has_ip_address", "has_suspicious_tld",
        "has_phishing_extension", "has_port", "has_double_slash_redirect",
        "has_hex_encoding", "num_at_symbols",
    }
    for k, v in features.items():
        if isinstance(v, float):
            rows.append({"Feature": k, "Value": f"{v:.4f}"})
        else:
            rows.append({"Feature": k, "Value": str(v)})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, height=420)


# ═════════════════════════════════════════════════════════════════════════════
# MODE: Single URL
# ═════════════════════════════════════════════════════════════════════════════
if mode == "Single URL":
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        url_input = st.text_input(
            "Enter a URL",
            placeholder="https://example.com/path?query=value",
            label_visibility="collapsed",
        )
    with col_btn:
        analyze = st.button("Analyze →")

    # Quick examples
    st.caption("Examples:")
    eg_cols = st.columns(4)
    examples = [
        ("✅ Google",       "https://www.google.com"),
        ("✅ GitHub",       "https://github.com/login"),
        ("🚨 Phishing #1",  "http://192.168.1.1/verify-account.php?user=john&token=abc123"),
        ("🚨 Phishing #2",  "http://paypal-secure-login.xyz/signin/confirm?sessid=%41%42"),
    ]
    for i, (label, url) in enumerate(examples):
        if eg_cols[i].button(label, key=f"eg{i}"):
            url_input = url
            analyze   = True

    if analyze and url_input.strip():
        with st.spinner("Analyzing …"):
            result = predictor.predict(url_input.strip())
            time.sleep(0.3)   # subtle UX delay

        render_risk_card(result)
        st.markdown(f"**URL analyzed:** `{result['url']}`")
        st.divider()

        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("#### Prediction summary")
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="metric-box"><div class="metric-value">{result["prob_phishing"]:.0%}</div><div class="metric-label">Phishing prob.</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-box"><div class="metric-value">{result["risk_level"]}</div><div class="metric-label">Risk level</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-box"><div class="metric-value">{result["confidence"]:.0%}</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)

            st.markdown("#### Key signals")
            feats = result["features"]
            flags = {
                "🔴 IP address in URL":       feats["has_ip_address"],
                "🔴 Suspicious keyword":      feats["has_suspicious_keyword"],
                "🔴 Suspicious TLD":          feats["has_suspicious_tld"],
                "🔴 Phishing file extension": feats["has_phishing_extension"],
                "🔴 Non-standard port":       feats["has_port"],
                "🔴 Double-slash redirect":   feats["has_double_slash_redirect"],
                "🔴 Hex encoding":            feats["has_hex_encoding"],
                "🔴 '@' symbol in URL":       feats["num_at_symbols"] > 0,
                "🟢 HTTPS":                   feats["has_https"],
            }
            for flag, active in flags.items():
                color = "#dc2626" if (active and not flag.startswith("🟢")) else ("#16a34a" if active else "#94a3b8")
                icon  = "●" if active else "○"
                st.markdown(
                    f'<span style="color:{color};font-size:0.95rem">{icon} {flag.split(" ",1)[1]}</span>',
                    unsafe_allow_html=True,
                )

        with col_b:
            st.markdown("#### Top numeric features")

            numeric_feats = {k: v for k, v in feats.items() if isinstance(v, (int, float)) and k not in {
                "has_https","has_ip_address","has_suspicious_keyword","has_suspicious_tld",
                "has_phishing_extension","has_port","has_double_slash_redirect","has_hex_encoding",
            }}
            # Normalize for bar chart
            top_keys = sorted(numeric_feats, key=lambda k: numeric_feats[k], reverse=True)[:12]
            vals = [numeric_feats[k] for k in top_keys]
            max_v = max(vals) if vals else 1

            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.barh(top_keys, vals, color="#2563eb", alpha=0.85)
            ax.set_xlabel("Value", fontsize=9)
            ax.tick_params(axis="y", labelsize=8)
            ax.tick_params(axis="x", labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.divider()
        with st.expander("Full feature table (all 30)"):
            render_feature_table(result["features"])

        with st.expander("Raw JSON output"):
            st.json(result)


# ═════════════════════════════════════════════════════════════════════════════
# MODE: Batch Analysis
# ═════════════════════════════════════════════════════════════════════════════
elif mode == "Batch Analysis":
    st.markdown("#### Batch URL Analysis")
    st.caption("Enter one URL per line (max 200).")

    default_urls = (
        "https://www.google.com\n"
        "https://github.com\n"
        "http://paypal-verify.xyz/login.php?user=abc\n"
        "http://192.168.0.1/account/confirm?token=%41%42\n"
        "https://www.amazon.com/dp/B08N5WRWNW\n"
        "http://secure-ebay-signin.tk/signin/?redirect=%2F\n"
    )
    raw = st.text_area("URLs", value=default_urls, height=200, label_visibility="collapsed")

    if st.button("Analyze all →"):
        urls = [u.strip() for u in raw.split("\n") if u.strip()]
        if not urls:
            st.warning("No URLs entered.")
        else:
            with st.spinner(f"Analyzing {len(urls)} URLs …"):
                results = predictor.predict_batch(urls)

            # Summary stats
            n_phish = sum(1 for r in results if r["is_phishing"])
            n_legit = len(results) - n_phish
            st.markdown("##### Summary")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total", len(results))
            s2.metric("Phishing 🚨", n_phish)
            s3.metric("Legitimate ✅", n_legit)
            s4.metric("Phish rate", f"{n_phish/len(results):.0%}")

            # Table
            df_results = pd.DataFrame([{
                "URL":         r["url"][:70] + "…" if len(r["url"]) > 70 else r["url"],
                "Label":       r["label"].upper(),
                "Risk":        r["risk_level"],
                "P(phishing)": f"{r['prob_phishing']:.1%}",
                "Confidence":  f"{r['confidence']:.1%}",
            } for r in results])

            def style_row(row):
                colors = {"SAFE": "#dcfce7", "LOW": "#fef9c3", "MEDIUM": "#ffedd5", "HIGH": "#fee2e2"}
                bg = colors.get(row["Risk"], "")
                return [f"background-color: {bg}"] * len(row)

            st.dataframe(
                df_results.style.apply(style_row, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            # Download
            csv = df_results.to_csv(index=False)
            st.download_button("Download CSV", csv, "phishing_results.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# MODE: Feature Explorer
# ═════════════════════════════════════════════════════════════════════════════
elif mode == "Feature Explorer":
    st.markdown("#### Feature Explorer")
    st.caption("See how any URL's 30 features compare against typical phishing vs. legitimate distributions.")

    url_fe = st.text_input("Enter a URL to explore", placeholder="https://example.com")

    if url_fe.strip():
        result = predictor.predict(url_fe.strip())
        feats  = result["features"]
        render_risk_card(result)

        st.divider()
        st.markdown("##### All 30 extracted features")

        rows = []
        BINARY = {
            "has_https","has_ip_address","has_suspicious_keyword","has_suspicious_tld",
            "has_phishing_extension","has_port","has_double_slash_redirect","has_hex_encoding",
        }
        for k, v in feats.items():
            ftype = "binary" if k in BINARY else "numeric"
            rows.append({"Feature": k, "Value": v, "Type": ftype})

        df_fe = pd.DataFrame(rows)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(6, 8))
            numeric_df = df_fe[df_fe["Type"] == "numeric"].copy()
            numeric_df = numeric_df.sort_values("Value", ascending=True)
            colors = ["#dc2626" if v > numeric_df["Value"].median() else "#2563eb"
                      for v in numeric_df["Value"]]
            ax.barh(numeric_df["Feature"], numeric_df["Value"], color=colors, alpha=0.85)
            ax.set_xlabel("Value", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title("Numeric features (red = above median)", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c2:
            st.markdown("**Binary features**")
            binary_df = df_fe[df_fe["Type"] == "binary"]
            for _, row in binary_df.iterrows():
                icon  = "🔴" if row["Value"] == 1 and row["Feature"] != "has_https" else ("🟢" if row["Feature"] == "has_https" and row["Value"] == 1 else "⚪")
                st.markdown(f"{icon} `{row['Feature']}` = **{int(row['Value'])}**")

        with st.expander("Raw feature dict"):
            st.json(feats)


# ═════════════════════════════════════════════════════════════════════════════
# MODE: About
# ═════════════════════════════════════════════════════════════════════════════
elif mode == "About":
    st.markdown("""
#### About this project

This tool uses a machine learning model trained on **30 URL-based features** to classify
URLs as phishing or legitimate in real time.

**Feature categories:**

| Category | Examples | Count |
|---|---|---|
| URL structure | `url_length`, `path_depth`, `num_subdomains` | 11 |
| Character patterns | `num_at_symbols`, `digit_ratio`, `num_hyphens` | 9 |
| Domain signals | `domain_entropy`, `has_suspicious_tld`, `has_ip_address` | 3 |
| Content signals | `has_suspicious_keyword`, `has_phishing_extension` | 3 |
| Encoding / obfuscation | `has_hex_encoding`, `has_double_slash_redirect` | 2 |
| Entropy & tokens | `url_entropy`, `token_count`, `longest_token_length` | 2 |

**Models trained:**
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- XGBoost *(if installed)*
- LightGBM *(if installed)*

The best-performing model (by F1 on validation set) is saved and used for inference.

**Disclaimer:** This tool uses structural URL features only — it does not fetch or
render the web page. It is a research/educational tool and should not be the sole
basis for security decisions.

**Quick start:**
```bash
pip install -r requirements.txt
python src/train_model.py
streamlit run app/streamlit_app.py
```

**API:**
```bash
uvicorn app.api:app --reload --port 8000
# Then visit http://localhost:8000/docs
```
    """)

    if predictor:
        st.divider()
        st.markdown("#### Live model info")
        st.json(predictor.meta)
