
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

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
    background-color: #0E1117;
}

h1, h2, h3 {
    text-align: center;
    color: white;
}

.result-box {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}

.probability-box {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}

.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}

div.stButton {
    text-align: center;
}

div.stButton > button:first-child {
    background-color: #ff4b4b;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 10px;
    padding: 12px 30px;
    border: none;
    width: 300px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🩺 Breast Cancer Prediction System")
st.markdown("### AI-Powered Medical Prediction App")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 About Project")

st.sidebar.info(
    """
This machine learning project predicts whether a tumor is:

- Benign
- Malignant

using breast cancer diagnostic features.
"""
)

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ ML Model Used")
st.sidebar.write("Support Vector Machine (SVM)")

st.sidebar.subheader("📊 Features Used")
st.sidebar.write("10 Important Medical Features")

st.sidebar.markdown("---")

st.sidebar.subheader("👨‍💻 Developed By")
st.sidebar.write("Subhas Chakraborty")

# ---------------- INPUT SECTION ----------------
st.markdown("## Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    radius_mean = st.slider("Radius Mean", 5.0, 30.0, 14.0)
    texture_mean = st.slider("Texture Mean", 5.0, 40.0, 19.0)
    perimeter_mean = st.slider("Perimeter Mean", 40.0, 200.0, 90.0)
    area_mean = st.slider("Area Mean", 100.0, 2500.0, 600.0)
    smoothness_mean = st.slider("Smoothness Mean", 0.05, 0.20, 0.10)

with col2:
    compactness_mean = st.slider("Compactness Mean", 0.01, 0.40, 0.10)
    concavity_mean = st.slider("Concavity Mean", 0.00, 0.50, 0.10)
    concave_points_mean = st.slider("Concave Points Mean", 0.00, 0.30, 0.05)
    symmetry_mean = st.slider("Symmetry Mean", 0.10, 0.40, 0.20)
    fractal_dimension_mean = st.slider("Fractal Dimension Mean", 0.04, 0.10, 0.06)

# ---------------- CREATE INPUT DATAFRAME ----------------
input_data = {
    'radius_mean': radius_mean,
    'texture_mean': texture_mean,
    'perimeter_mean': perimeter_mean,
    'area_mean': area_mean,
    'smoothness_mean': smoothness_mean,
    'compactness_mean': compactness_mean,
    'concavity_mean': concavity_mean,
    'concave points_mean': concave_points_mean,
    'symmetry_mean': symmetry_mean,
    'fractal_dimension_mean': fractal_dimension_mean
}

input_df = pd.DataFrame([input_data])

# ---------------- ENSURE COLUMN ORDER ----------------
input_df = input_df[expected_columns]

# ---------------- SCALE INPUT ----------------
input_scaled = scaler.transform(input_df)

# ---------------- PREDICTION BUTTON ----------------
if st.button("Predict Cancer Type"):

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    st.markdown("---")

    # ---------------- RESULT ----------------
    st.markdown("# Prediction Result")

    if prediction[0] == 1:
        st.markdown(
            '<div class="result-box" style="background-color:#4B0000;color:white;">⚠️ Malignant Tumor Detected</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-box" style="background-color:#014421;color:white;">✅ Benign Tumor Detected</div>',
            unsafe_allow_html=True
        )

    # ---------------- METRICS ----------------
    st.markdown("# Prediction Probability")

    col3, col4 = st.columns(2)

    with col3:
        st.metric(
            label="Benign Probability",
            value=f"{probability[0][0]*100:.2f}%"
        )

    with col4:
        st.metric(
            label="Malignant Probability",
            value=f"{probability[0][1]*100:.2f}%"
        )

    # ---------------- CHARTS ----------------
    st.markdown("# 📊 Visual Analysis")

    # Probability Chart
    prob_df = pd.DataFrame({
        'Cancer Type': ['Benign', 'Malignant'],
        'Probability': [
            probability[0][0]*100,
            probability[0][1]*100
        ]
    })

    fig_bar = px.bar(
        prob_df,
        x='Cancer Type',
        y='Probability',
        text='Probability',
        title='Prediction Probability Analysis'
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # Radar Chart
    categories = list(input_data.keys())
    values = list(input_data.values())

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Patient Data'
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            )
        ),
        showlegend=True,
        title='Patient Feature Radar Chart'
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    # Feature Table
    st.markdown("# 📋 Patient Input Summary")

    st.dataframe(input_df)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<h4 style='text-align:center;'>Made with ❤️ using Streamlit & Machine Learning</h4>",
    unsafe_allow_html=True
)
