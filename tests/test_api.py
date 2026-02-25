import requests
import json

# API Endpoint
URL = "http://127.0.0.1:8000/predict"

# -----------------------------
# Test Cases
# -----------------------------

test_cases = [
    {
        "name": "Test Case 1 - Respiratory Case",
        "data": {
            "Fever": "Yes",
            "Cough": "Yes",
            "Fatigue": "Yes",
            "Difficulty_Breathing": "Yes",
            "Age": 45,
            "Gender": "Male",
            "Blood_Pressure": "Normal",
            "Cholesterol_Level": "Normal"
        }
    },
    {
        "name": "Test Case 2 - Elderly High Risk",
        "data": {
            "Fever": "Yes",
            "Cough": "Yes",
            "Fatigue": "Yes",
            "Difficulty_Breathing": "Yes",
            "Age": 72,
            "Gender": "Female",
            "Blood_Pressure": "High",
            "Cholesterol_Level": "High"
        }
    },
    {
        "name": "Test Case 3 - Mild Viral",
        "data": {
            "Fever": "No",
            "Cough": "Yes",
            "Fatigue": "Yes",
            "Difficulty_Breathing": "No",
            "Age": 25,
            "Gender": "Female",
            "Blood_Pressure": "Normal",
            "Cholesterol_Level": "Normal"
        }
    },
    {
        "name": "Test Case 4 - Cardio Risk Profile",
        "data": {
            "Fever": "No",
            "Cough": "No",
            "Fatigue": "Yes",
            "Difficulty_Breathing": "Yes",
            "Age": 60,
            "Gender": "Male",
            "Blood_Pressure": "High",
            "Cholesterol_Level": "High"
        }
    },
    {
        "name": "Test Case 5 - Elderly High Risk",
        "data": {
            "Fever": "Yes",
            "Cough": "Yes",
            "Fatigue": "Yes",
            "Difficulty_Breathing": "Yes",
            "Age": 72,
            "Gender": "Female",
            "Blood_Pressure": "High",
            "Cholesterol_Level": "High"
        }
    },,
    {
        "name": "Test Case 6 - Elderly High Risk",
        "data": {
            "Fever": "Yes",
            "Cough": "Yes",
            "Fatigue": "Yes",
            "Difficulty_Breathing": "Yes",
            "Age": 72,
            "Gender": "Female",
            "Blood_Pressure": "High",
            "Cholesterol_Level": "High"
        }
    }
]

# -----------------------------
# Run Tests
# -----------------------------

for test in test_cases:
    print("\n" + "=" * 60)
    print(test["name"])
    print("=" * 60)

    try:
        response = requests.post(URL, json=test["data"])

        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=4))
        else:
            print(f"Error {response.status_code}: {response.text}")

    except Exception as e:
        print("Request failed:", str(e))