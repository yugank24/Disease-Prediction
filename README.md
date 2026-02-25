🩺 AI Symptom Checker API

Machine Learning–based Disease Prediction API built using FastAPI and XGBoost.

This system predicts the Top 3 most probable diseases based on patient symptoms and provides risk assessment with recommendations.


📌 Features
1. Top 3 disease predictions with confidence %
2. Clinical risk assessment (Low / Medium / High)
3. Rule-based override logic
4. REST API using FastAPI
5. XGBoost ML model
6. Structured JSON output


🏗️ Architecture Overview
Client (Postman / UI)
        ↓
FastAPI REST API
        ↓
Input Validation (Pydantic)
        ↓
Preprocessing Layer
        ↓
XGBoost ML Model
        ↓
Risk Rule Engine
        ↓
Recommendation Engine
        ↓
Explainability Module
        ↓
JSON Response


📂 Project Structure
symptom-checker/
│
├── app/
│   └── main.py              # FastAPI application
│
├── data/
│   └── dataset.csv          # Training dataset
│
├── model/
│   ├── train.py             # Model training script
│   └── disease_model.pkl    # Saved trained model
│   └── label_encoder.pkl
│
├── tests/
│   └── test_api.py          # API unit tests
│
├── requirements.txt
└── README.md


📊 Dataset Description
Feature	Description
Fever	Yes / No
Cough	Yes / No
Fatigue	Yes / No
Difficulty Breathing	Yes / No
Age	Integer (years)
Gender	Male / Female
Blood Pressure	Low / Normal / High
Cholesterol Level	Normal / High
Outcome Variable	Positive / Negative

Categorical values are encoded numerically during preprocessing.


🤖 Machine Learning Model

Algorithm: XGBoost Classifier
Problem Type: Binary classification
Target Variable: Disease outcome (Positive / Negative)

Why XGBoost?
Excellent performance on tabular clinical data
Handles non-linear feature interactions
Robust and widely used in production ML systems


🧠 Hybrid Intelligence Approach
This system uses two layers of intelligence:

1️⃣ ML-Based Prediction
1. Predicts disease probability using trained XGBoost model

2️⃣ Rule-Based Risk Engine
Overrides ML predictions in clinically high-risk scenarios, such as:
1. Age > 60 with breathing difficulty
2. High blood pressure and high cholesterol

🏥 Risk Assessment Logic
Probability Based
Probability	Risk Level
> 0.75	High
0.45–0.75	Medium
< 0.45	Low


🔌 API Usage
Endpoint: POST /predict

Sample Request
{
  "Fever": "Yes",
  "Cough": "Yes",
  "Fatigue": "Yes",
  "Difficulty_Breathing": "Yes",
  "Age": 65,
  "Gender": "Male",
  "Blood_Pressure": "High",
  "Cholesterol_Level": "High"
}

Sample Response
{
  "prediction": {
    "disease_risk_probability": 0.82,
    "confidence_percentage": 82.0
  },
  "risk_assessment": {
    "risk_level": "High",
    "recommended_action": "Consult Doctor Immediately"
  },
  "clinical_flags": [
    "Fever detected",
    "Breathing difficulty reported",
    "High-risk age group",
    "Elevated blood pressure",
    "High cholesterol level"
  ]
}


🧪 Testing

Run API tests using: pytest tests/test_api.py

▶️ Running the Application
    1️⃣ Install Dependencies
        pip install -r requirements.txt
    2️⃣ Train the Model
        python model/train.py
    3️⃣ Start API Server
        uvicorn app.main:app --reload
    4️⃣ Open API Docs
        http://127.0.0.1:8000/docs


🛠️ Technologies Used

Python
FastAPI
XGBoost
Scikit-learn
Pandas
NumPy
Joblib
Pytest


🔒 Design Considerations

Deterministic ML preferred over LLMs for structured clinical data
Explainability prioritized via clinical flags
Clear separation between ML prediction and medical logic
Production-ready API structure


🚀 Future Enhancements
SHAP-based model explainability
Dockerization
Logging and monitoring
Multi-disease prediction
Model versioning
