import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

df = pd.read_csv("Disease_predictor.csv")

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("✅ All Files Saved Successfully")