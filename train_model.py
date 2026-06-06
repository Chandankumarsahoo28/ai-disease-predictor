import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Disease_predictor.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Disease_predictor.csv not found in the same folder as train_model.py")

df = pd.read_csv(DATA_PATH)

if "prognosis" not in df.columns:
    raise ValueError("Dataset must contain a 'prognosis' column")

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(BASE_DIR, "encoder.pkl"), "wb") as f:
    pickle.dump(encoder, f)

with open(os.path.join(BASE_DIR, "columns.pkl"), "wb") as f:
    pickle.dump(list(X.columns), f)

print("✅ model.pkl, encoder.pkl, columns.pkl saved successfully")
