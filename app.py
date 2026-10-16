"""Streamlit deployment (Self-study: Model Deployment).  Run:  streamlit run app.py"""
import joblib, pandas as pd, streamlit as st
from src.features import extract

st.set_page_config(page_title="Phishing URL Detector", page_icon="🎣")
st.title("🎣 Phishing Website Detector")
st.caption("AIML mini-project - lexical URL features + ML classifier. The URL is analysed as text only; it is never visited.")

@st.cache_resource
def load(): return joblib.load("models/best_model.joblib")
bundle = load(); model, names = bundle["model"], bundle["features"]

url = st.text_input("Enter a URL", placeholder="http://paypal.secure-login.xyz/verify/account")
if st.button("Check URL") and url:
    feats = pd.DataFrame([extract(url)])[names]
    p = float(model.predict_proba(feats.values)[0, 1])
    st.progress(p); st.subheader(f"Phishing probability: {p:.1%}")
    (st.error if p >= .5 else st.success)("⚠️ Likely PHISHING" if p >= .5 else "✅ Likely legitimate")
    with st.expander("Extracted features"): st.dataframe(feats.T.rename(columns={0: "value"}))
    st.caption(f"Model: {bundle['name']}. Lexical models can be wrong - never rely on one signal alone.")
