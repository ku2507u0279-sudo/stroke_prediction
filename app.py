import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🩺", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "stroke_model.joblib"
THRESHOLD_PATH = BASE_DIR / "threshold.json"

FEATURES = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]

st.markdown("""
<style>
.stApp { background: #f4f7fb; color: #172033; }
h1, h2, h3, p, label { color: #172033 !important; }
.hero { padding: 1.4rem 1.6rem; border-radius: 16px; background: linear-gradient(120deg,#e7f0ff,#f7fbff); border: 1px solid #cfe0fb; margin-bottom: 1rem; }
.hero h1 { color: #12305a !important; margin: 0 0 .35rem 0; }
.hero p { color: #334b6b !important; margin: 0; }
div[data-testid="stMetric"] { background: #fff; border: 1px solid #d8e2ef; border-radius: 14px; padding: 1rem; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stCaptionContainer"], .stCaption { color: #172033 !important; opacity: 1 !important; }
.stButton > button, [data-testid="stFormSubmitButton"] button { width: 100%; background: #1769d2; color: white !important; border-radius: 10px; border: 0; font-weight: 700; padding: .7rem; }
.note { color: #475569 !important; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("stroke_model.joblib was not found in the repository root.")
    return joblib.load(MODEL_PATH)

def load_cutoff():
    if not THRESHOLD_PATH.exists():
        return 0.5
    try:
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, (int, float)):
            cutoff = float(data)
        elif isinstance(data, dict):
            cutoff = next((float(data[k]) for k in ("threshold", "cutoff", "best_threshold", "classification_cutoff") if k in data), 0.5)
        else:
            cutoff = 0.5
        return cutoff if 0 <= cutoff <= 1 else 0.5
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0.5

st.markdown("""
<div class="hero">
  <h1>🩺 Stroke Risk Predictor</h1>
  <p>Enter patient details to get a machine-learning model estimate.</p>
</div>
""", unsafe_allow_html=True)

st.info("For educational screening support only. This tool does not diagnose stroke or replace professional medical advice.")
st.subheader("Patient details")

with st.form("stroke_prediction_form"):
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        heart_disease = st.selectbox("Heart disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with right:
        avg_glucose_level = st.number_input("Average glucose level", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    submitted = st.form_submit_button("Estimate stroke risk")

if submitted:
    if age <= 0 or avg_glucose_level <= 0 or bmi <= 0:
        st.error("Please enter values greater than zero for age, glucose level, and BMI.")
    else:
        try:
            model = load_model()
            cutoff = load_cutoff()
            row = pd.DataFrame([[age, hypertension, heart_disease, avg_glucose_level, bmi]], columns=FEATURES)
            probabilities = model.predict_proba(row)
            classes = list(getattr(model, "classes_", []))
            stroke_index = classes.index(1) if 1 in classes else (1 if probabilities.shape[1] > 1 else 0)
            score = float(probabilities[0][stroke_index])
            flagged = score >= cutoff

            st.divider()
            st.subheader("Model Output")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Stroke Risk Score", f"{score * 100:.1f}%")
            with col2:
                st.metric("Classification", "Flagged" if flagged else "Not flagged")

            st.progress(min(max(score, 0.0), 1.0))
            st.caption(f"Classification cutoff: {cutoff:.3f}")

            if flagged:
                st.error("Higher model score: this input was flagged by the selected cutoff. Please discuss health concerns with a healthcare professional.")
            else:
                st.success("No high-risk flag detected. This does not rule out stroke risk.")

            st.caption("This score is a model output, not necessarily a clinically calibrated probability. Do not use it to make medical decisions.")
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error("Prediction failed. Check that the model file and scikit-learn version match the training environment.")
            st.code(f"{type(e).__name__}: {e}")
else:
    st.caption("Fill in the fields and click Estimate stroke risk to see the result.")
