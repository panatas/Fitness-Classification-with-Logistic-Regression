
import pickle
import streamlit as st
import pandas as pd

# -----------------------------
# 1. Load the trained model
# -----------------------------
@st.cache_resource
def load_model():
    with open("fitness_model.pkl", "rb") as file:
        bundle = pickle.load(file)
    return bundle

bundle = load_model()
model = bundle["model"]
feature_names = bundle["feature_names"]

# -----------------------------
# 2. Page configuration
# -----------------------------
st.set_page_config(
    page_title="Fitness Prediction",
    page_icon="💪",
    layout="centered"
)

st.title("💪 Fitness Prediction")
st.write(
    "Enter your information below to explore "
    "the prediction from our Logistic Regression model."
)

st.info(
    "This is a machine-learning demonstration, "
    "not a medical or fitness assessment."
)

# -----------------------------
# 3. User input
# -----------------------------
st.header("Your Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (years)", min_value=1, max_value=100, value=25
    )
    height_cm = st.number_input(
        "Height (cm)", min_value=50.0, max_value=250.0,
        value=170.0
    )
    weight_kg = st.number_input(
        "Weight (kg)", min_value=10.0, max_value=300.0,
        value=65.0
    )
    heart_rate = st.number_input(
        "Heart Rate (bpm)", min_value=30, max_value=220,
        value=75
    )

with col2:
    blood_pressure = st.number_input(
        "Blood Pressure (mmHg)", min_value=40, max_value=250,
        value=120
    )
    sleep_hours = st.number_input(
        "Sleep Hours", min_value=0.0, max_value=24.0,
        value=7.0, step=0.5
    )
    nutrition_quality = st.slider(
        "Nutrition Quality (1–10)", 1, 10, 7
    )
    activity_index = st.slider(
        "Activity Index (1–10)", 1, 10, 5
    )

gender = st.selectbox("Gender", ["Male", "Female"])
smokes = st.selectbox("Do you smoke?", ["No", "Yes"])

# Match the encoding used in the notebook:
# Male = 0, Female = 1
# No = 0, Yes = 1
user_data = pd.DataFrame([{
    "age": age,
    "height_cm": height_cm,
    "weight_kg": weight_kg,
    "heart_rate": heart_rate,
    "blood_pressure": blood_pressure,
    "sleep_hours": sleep_hours,
    "nutrition_quality": nutrition_quality,
    "activity_index": activity_index,
    "smokes": 1 if smokes == "Yes" else 0,
    "gender": 1 if gender == "Female" else 0
}])

# Preserve the exact feature order used during training
user_data = user_data[feature_names]

# -----------------------------
# 4. Make a prediction
# -----------------------------
if st.button("Predict Fitness", type="primary"):
    probability = float(model.predict(user_data).iloc[0])
    prediction = int(probability >= 0.5)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("The model predicts: Fit (Class 1)")
    else:
        st.warning("The model predicts: Not Fit (Class 0)")

    st.metric(
        "Model score for Class 1",
        f"{probability * 100:.2f}%"
    )

    st.caption(
        "This score comes from the model's estimated probability. "
        "It is not a medical measurement or a guarantee of fitness."
    )

# -----------------------------
# 5. Show test-set performance
# -----------------------------
st.divider()
st.header("Model Performance")

st.write("Evaluation results saved from your notebook:")

metrics = {
    "Accuracy": bundle["test_accuracy"],
    "Precision": bundle["test_precision"],
    "Recall": bundle["test_recall"],
    "F1-score": bundle["test_f1"]
}

metric_cols = st.columns(4)

for col, (name, value) in zip(metric_cols, metrics.items()):
    col.metric(name, f"{value:.2f}%")
