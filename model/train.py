import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


# ---------------------------
# Load Dataset
# ---------------------------
DATA_PATH = "data/dataset.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Total rows:", len(df))


# ---------------------------
# Encoding Mappings
# ---------------------------
binary_map = {"Yes": 1, "No": 0}
gender_map = {"Male": 0, "Female": 1}
bp_map = {"Low": 0, "Normal": 1, "High": 2}
chol_map = {"Normal": 0, "High": 1}

df["Fever"] = df["Fever"].map(binary_map)
df["Cough"] = df["Cough"].map(binary_map)
df["Fatigue"] = df["Fatigue"].map(binary_map)
df["Difficulty Breathing"] = df["Difficulty Breathing"].map(binary_map)
df["Gender"] = df["Gender"].map(gender_map)
df["Blood Pressure"] = df["Blood Pressure"].map(bp_map)
df["Cholesterol Level"] = df["Cholesterol Level"].map(chol_map)

df = df.dropna().reset_index(drop=True)


# ---------------------------
# Remove Rare Diseases (<2 samples)
# ---------------------------
class_counts = df["Disease"].value_counts()
valid_diseases = class_counts[class_counts >= 2].index

df = df[df["Disease"].isin(valid_diseases)].reset_index(drop=True)

print("Remaining diseases after filtering:", df["Disease"].nunique())
print("Remaining rows:", len(df))


# ---------------------------
# Encode Target (AFTER filtering)
# ---------------------------
label_encoder = LabelEncoder()
df["Disease"] = label_encoder.fit_transform(df["Disease"])

print("Final unique diseases:", len(label_encoder.classes_))


# ---------------------------
# Features & Target
# ---------------------------
X = df.drop(columns=["Disease", "Outcome Variable"], errors="ignore")
y = df["Disease"]

print("Min label:", y.min())
print("Max label:", y.max())
print("Unique label count:", len(y.unique()))


# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ---------------------------
# Model
# ---------------------------
model = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)


# ---------------------------
# Evaluation
# ---------------------------
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ---------------------------
# Save Model & Encoder
# ---------------------------
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/disease_model.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")

print("\nModel and label encoder saved successfully!")
print("Training completed.")
