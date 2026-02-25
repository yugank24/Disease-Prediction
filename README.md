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



📊 Dataset Description
1. Feature	Description
2. Fever	Yes / No
3. Cough	Yes / No
4. Fatigue	Yes / No
5. Difficulty Breathing	Yes / No
6. Age	Integer (years)
7. Gender	Male / Female
8. Blood Pressure	Low / Normal / High
9. Cholesterol Level	Normal / High
10. Outcome Variable	Positive / Negative

Categorical values are encoded numerically during preprocessing.


🤖 Machine Learning Model

1. Algorithm: XGBoost Classifier
2. Problem Type: Binary classification
3. Target Variable: Disease outcome (Positive / Negative)

Why XGBoost?
1. Excellent performance on tabular clinical data
2. Handles non-linear feature interactions
3. Robust and widely used in production ML systems


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
1. > 0.75	High
2. 0.45–0.75	Medium
2. < 0.45	Low


🔌 API Usage
Endpoint: POST /predict

Sample Request: 
{
  "Fever": "No",
  "Cough": "No",
  "Fatigue": "Yes",
  "Difficulty_Breathing": "Yes",
  "Age": 60,
  "Gender": "Male",
  "Blood_Pressure": "High",
  "Cholesterol_Level": "High"
}

Sample Response: 
{
  "prediction": {
    "top_3_predictions": [
      {
        "disease": "Bronchitis",
        "confidence_percentage": 61.35
      },
      {
        "disease": "Osteoporosis",
        "confidence_percentage": 19.16
      },
      {
        "disease": "Rheumatoid Arthritis",
        "confidence_percentage": 4.95
      }
    ]
  },
  "risk_assessment": {
    "risk_level": "High",
    "recommended_action": "Consult Doctor Immediately"
  },
  "clinical_flags": [
    "Breathing difficulty reported",
    "Elevated blood pressure",
    "High cholesterol level"
  ]
}


🧪 Testing

Run API tests using: pytest tests/test_api.py

▶️ Running the Application
    1️ Install Dependencies
        pip install -r requirements.txt
    2️ Train the Model
        python model/train.py
    3️ Start API Server
        uvicorn app.main:app --reload
    4️ Open API Docs
        http://127.0.0.1:8000/docs


🛠️ Technologies Used

1. Python
2. FastAPI
3. XGBoost
4. Scikit-learn
5. Pandas
6. NumPy
7. Joblib
8. Pytest


🔒 Design Considerations

1. Deterministic ML preferred over LLMs for structured clinical data
2. Explainability prioritized via clinical flags
3. Clear separation between ML prediction and medical logic
4. Production-ready API structure


🚀 Future Enhancements
1. SHAP-based model explainability
2. Dockerization
3. Logging and monitoring
4. Multi-disease prediction
5. Model versioning
