import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

h1 {
    text-align: center;
    color: #ff4b4b;
}

.stButton>button {
    width: 100%;
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🩺 Breast Cancer Predictor")

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ ML Model Used")
st.sidebar.write("Support Vector Machine (SVM)")

st.sidebar.subheader("🧠 Explainable AI")
st.sidebar.write("SHAP Explainability")

st.sidebar.subheader("📊 Features Used")
st.sidebar.write("10 Medical Features")

st.sidebar.markdown("---")

st.sidebar.subheader("👨‍💻 Developed By")
st.sidebar.markdown(
    "<div style='font-size:18px; font-weight:bold;'>Subhas Chakraborty</div>",
    unsafe_allow_html=True
)

# ---------------- TITLE ----------------
st.title("🩺 Breast Cancer Prediction System")

st.markdown(
    "<h4 style='text-align:center;'>Predict whether the tumor is Benign or Malignant</h4>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("Radius Mean", 0.0, 50.0, 14.0)
    texture_mean = st.number_input("Texture Mean", 0.0, 50.0, 20.0)
    perimeter_mean = st.number_input("Perimeter Mean", 0.0, 200.0, 90.0)
    area_mean = st.number_input("Area Mean", 0.0, 3000.0, 600.0)
    smoothness_mean = st.number_input("Smoothness Mean", 0.0, 1.0, 0.1)

with col2:
    compactness_mean = st.number_input("Compactness Mean", 0.0, 1.0, 0.1)
    concavity_mean = st.number_input("Concavity Mean", 0.0, 1.0, 0.1)
    concave_points_mean = st.number_input("Concave Points Mean", 0.0, 1.0, 0.05)
    symmetry_mean = st.number_input("Symmetry Mean", 0.0, 1.0, 0.2)
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", 0.0, 1.0, 0.06)

# ---------------- PREDICT BUTTON ----------------
st.markdown("<br>", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])

with col_btn2:
    predict_button = st.button("🔍 Predict Cancer Type")

# ---------------- PREDICTION ----------------
if predict_button:

    input_data = pd.DataFrame({
        'radius_mean': [radius_mean],
        'texture_mean': [texture_mean],
        'perimeter_mean': [perimeter_mean],
        'area_mean': [area_mean],
        'smoothness_mean': [smoothness_mean],
        'compactness_mean': [compactness_mean],
        'concavity_mean': [concavity_mean],
        'concave points_mean': [concave_points_mean],
        'symmetry_mean': [symmetry_mean],
        'fractal_dimension_mean': [fractal_dimension_mean]
    })

    # Add missing columns
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[expected_columns]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.markdown("---")

    # RESULT
    if prediction == 1:
        st.markdown(
            "<div class='result-box' style='background-color:#ffcccc; color:#b30000;'>⚠️ Malignant Tumor Detected</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-box' style='background-color:#ccffcc; color:#006600;'>✅ Benign Tumor Detected</div>",
            unsafe_allow_html=True
        )

    # Probability Chart
    st.subheader("📊 Prediction Probability")

    prob_df = pd.DataFrame({
        "Class": ["Benign", "Malignant"],
        "Probability": probability
    })

    st.bar_chart(prob_df.set_index("Class"))

    # ---------------- SHAP EXPLAINABILITY ----------------
    st.subheader("🧠 Explainable AI with SHAP")

    explainer = shap.KernelExplainer(
        model.predict_proba,
        input_scaled
    )

    shap_values = explainer.shap_values(input_scaled)

    st.write("Feature Impact on Prediction")

    fig, ax = plt.subplots()

    shap.summary_plot(
        shap_values,
        input_scaled,
        feature_names=expected_columns,
        show=False
    )

    st.pyplot(fig)

    st.success("Prediction Completed Successfully ✅")