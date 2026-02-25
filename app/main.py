from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="AI Symptom Checker API",
    description="Multi-class disease prediction with rule-based risk stratification",
    version="2.0"
)

# -----------------------------
# Load Model & Encoder
# -----------------------------
try:
    model = joblib.load("model/disease_model.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")


# -----------------------------
# Input Schema
# -----------------------------
class PatientInput(BaseModel):
    Fever: str
    Cough: str
    Fatigue: str
    Difficulty_Breathing: str
    Age: int
    Gender: str
    Blood_Pressure: str
    Cholesterol_Level: str


# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess(data: PatientInput):

    binary_map = {"Yes": 1, "No": 0}
    gender_map = {"Male": 0, "Female": 1}
    bp_map = {"Low": 0, "Normal": 1, "High": 2}
    chol_map = {"Normal": 0, "High": 1}

    try:
        features = np.array([[
            binary_map[data.Fever],
            binary_map[data.Cough],
            binary_map[data.Fatigue],
            binary_map[data.Difficulty_Breathing],
            data.Age,
            gender_map[data.Gender],
            bp_map[data.Blood_Pressure],
            chol_map[data.Cholesterol_Level]
        ]])
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail="Invalid categorical input provided."
        )

    return features


# -----------------------------
# Risk Calculation Logic
# -----------------------------
def calculate_risk(top_confidence, data):

    if top_confidence > 70:
        base_risk = "High"
    elif top_confidence > 40:
        base_risk = "Medium"
    else:
        base_risk = "Low"

    # Clinical overrides
    if data.Age > 60 and data.Difficulty_Breathing == "Yes":
        return "High"

    if data.Blood_Pressure == "High" and data.Cholesterol_Level == "High":
        return "High"

    return base_risk


# -----------------------------
# Recommendation Engine
# -----------------------------
def recommend_action(risk_level):

    if risk_level == "High":
        return "Consult Doctor Immediately"
    elif risk_level == "Medium":
        return "Schedule Doctor Consultation"
    else:
        return "Home Care and Monitor Symptoms"


# -----------------------------
# Clinical Explanation Flags
# -----------------------------
def generate_explanation(data):

    reasons = []

    if data.Fever == "Yes":
        reasons.append("Fever detected")
    if data.Difficulty_Breathing == "Yes":
        reasons.append("Breathing difficulty reported")
    if data.Age > 60:
        reasons.append("High-risk age group")
    if data.Blood_Pressure == "High":
        reasons.append("Elevated blood pressure")
    if data.Cholesterol_Level == "High":
        reasons.append("High cholesterol level")

    return reasons


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: PatientInput):

    processed = preprocess(data)

    probabilities = model.predict_proba(processed)[0]

    # Get Top 3 diseases
    top3_indices = np.argsort(probabilities)[-3:][::-1]

    top3_predictions = []

    for idx in top3_indices:
        disease_name = label_encoder.inverse_transform([idx])[0]
        confidence = round(float(probabilities[idx] * 100), 2)

        top3_predictions.append({
            "disease": disease_name,
            "confidence_percentage": confidence
        })

    # Highest confidence used for risk logic
    highest_confidence = top3_predictions[0]["confidence_percentage"]

    risk = calculate_risk(highest_confidence, data)
    action = recommend_action(risk)
    explanation = generate_explanation(data)

    return {
        "prediction": {
            "top_3_predictions": top3_predictions
        },
        "risk_assessment": {
            "risk_level": risk,
            "recommended_action": action
        },
        "clinical_flags": explanation
    }
