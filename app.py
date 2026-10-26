import os, json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🩺", layout="centered")
st.markdown("""
<style>
.stApp{background:linear-gradient(180deg,#f1f8fc 0%,#ffffff 58%)}
.block-container{max-width:850px;padding-top:2rem;padding-bottom:3rem}
.hero{padding:1.5rem 1.6rem;border-radius:20px;background:linear-gradient(120deg,#123a63,#167d9a);color:white;margin-bottom:1rem}
.hero h1{color:white;margin:0;font-size:2.1rem}.hero p{color:#e5f5ff;margin:.45rem 0 0}
.hint{background:#eef7fb;border-left:4px solid #1683a5;padding:.8rem 1rem;border-radius:8px;color:#24465b}
div[data-testid="stMetric"]{background:white;border:1px solid #dbe8f0;padding:1rem;border-radius:14px}
div.stButton>button{width:100%;border-radius:12px;height:3rem;font-weight:700}
</style>
""",unsafe_allow_html=True)

HERE=os.path.dirname(__file__)
MODEL_PATH=os.path.join(HERE,"stroke_model.joblib")
THRESHOLD_PATH=os.path.join(HERE,"threshold.json")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def load_threshold():
    try:
        with open(THRESHOLD_PATH,"r") as f:
            return float(json.load(f)["threshold"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return 0.75

st.markdown('<div class="hero"><h1>🩺 Stroke Risk Predictor</h1><p>Simple, interactive machine-learning demonstration.</p></div>',unsafe_allow_html=True)
st.markdown('<div class="hint">Enter five details and click <b>Estimate</b>. The tuned threshold aims to reduce false-positive flags compared with the default 0.50 cutoff.</div>',unsafe_allow_html=True)

if not os.path.exists(MODEL_PATH):
    st.error("Missing stroke_model.joblib. Upload it beside app.py in your GitHub repository.")
    st.stop()

with st.form("patient_form"):
    st.subheader("Patient details")
    left,right=st.columns(2)
    with left:
        age=st.slider("Age (years)",1,100,45)
        hypertension=st.selectbox("Hypertension",["No","Yes"])
        bmi=st.number_input("BMI",min_value=10.0,max_value=80.0,value=27.0,step=0.1)
    with right:
        heart_disease=st.selectbox("Heart disease",["No","Yes"])
        glucose=st.number_input("Average glucose level",min_value=40.0,max_value=300.0,value=100.0,step=1.0)
    submitted=st.form_submit_button("Estimate",type="primary")

if submitted:
    row=pd.DataFrame([{
        "age":float(age),
        "hypertension":int(hypertension=="Yes"),
        "heart_disease":int(heart_disease=="Yes"),
        "avg_glucose_level":float(glucose),
        "bmi":float(bmi)
    }])
    score=float(load_model().predict_proba(row)[0][1])
    threshold=load_threshold()
    flagged=score>=threshold
    st.divider()
    st.subheader("Model output")
    c1,c2=st.columns(2)
    c1.metric("Model score",f"{score*100:.1f}%")
    c2.metric("Classification","Flagged" if flagged else "Not flagged")
    st.progress(min(max(score,0.0),1.0))
    st.caption(f"Classification cutoff selected using validation data: {threshold:.3f}")
    if flagged:
        st.warning("The model flagged this input. This is not a diagnosis; please discuss health concerns with a qualified healthcare professional.")
    else:
        st.success("The model did not flag this input at the selected cutoff. This does not rule out stroke risk.")
    st.caption("The score is not a clinically calibrated probability and must not be used for medical decisions.")

st.divider()
st.markdown("""**About this demo**  
Random Forest model using age, hypertension, heart disease, average glucose level, and BMI.

**Medical disclaimer:** This classroom project is not a diagnostic or screening tool. Sudden face drooping, arm weakness, or speech difficulty requires emergency medical care.""")
