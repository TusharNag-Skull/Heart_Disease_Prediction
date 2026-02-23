import streamlit as st
import pandas as pd
import mlflow
import os
import mlflow.sklearn
from dotenv import load_dotenv
load_dotenv()

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


mlflow.set_tracking_uri(
    "https://dagshub.com/TusharNag-Skull/Heart_Disease_Prediction.mlflow"
)


# Load model from MLflow Registry (Production)
@st.cache_resource
def load_model():
    return mlflow.sklearn.load_model(model_uri="models:/HeartDiseaseModel@production")
        

model = load_model()


# Streamlit UI

st.set_page_config(page_title="Heart Disease Prediction",layout="centered")

st.title("Heart Disease Prediction")
st.write("Enter patient clinical data to predict the likelihood of heart disease")


col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=20, max_value=100, value=50)
    BP = st.number_input("Blood Pressure (BP)", min_value=80, max_value=200, value=120)
    Cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)

with col2:
    Max_HR = st.number_input("Max Heart Rate (Max HR)", min_value=60, max_value=210, value=150)
    ST_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
    Sex = st.selectbox("Sex", ["Male", "Female"])

Chest_pain_type = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
FBS_over_120 = st.selectbox("FBS over 120", [0, 1])
EKG_results = st.selectbox("EKG Results", [0, 1, 2])
Exercise_angina = st.selectbox("Exercise Angina", ["Yes", "No"])
Slope_of_ST = st.selectbox("Slope of ST", [1, 2, 3])
Number_of_vessels_fluro = st.selectbox("Number of Vessels (Fluoroscopy)", [0, 1, 2, 3])
Thallium = st.selectbox("Thallium", [3, 6, 7])


# Encode categorical values
Sex = 1 if Sex == "Male" else 0
Exercise_angina = 1 if Exercise_angina == "Yes" else 0


# Prediction
if st.button("Predict", use_container_width=True):

    input_df = pd.DataFrame([{
        "Age": Age,
        "Sex": Sex,
        "Chest pain type": Chest_pain_type,
        "BP": BP,
        "Cholesterol": Cholesterol,
        "FBS over 120": FBS_over_120,
        "EKG results": EKG_results,
        "Max HR": Max_HR,
        "Exercise angina": Exercise_angina,
        "ST depression": ST_depression,
        "Slope of ST": Slope_of_ST,
        "Number of vessels fluro": Number_of_vessels_fluro,
        "Thallium": Thallium
    }])

    # Class prediction
    prediction = model.predict(input_df)[0]

    # Probability prediction (safe handling)
    probability = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)
        probability = proba[0][1] if proba.shape[1] > 1 else proba[0][0]

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if int(prediction) == 1:
            st.error("Heart Disease Likely")
        else:
            st.success("No Heart Disease Detected")

    with col2:
        if probability is not None:
            st.metric("Confidence", f"{probability:.2%}")

    if probability is not None:
        st.info(f"**Probability of Heart Disease: {probability:.4f}**")

st.caption("Model served from MLflow Model Registry (production)")
