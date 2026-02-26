import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Generate Dataset

np.random.seed(42)

n_samples = 5000

data = pd.DataFrame({
    "age": np.random.randint(18, 40, n_samples),
    "daily_hours": np.random.randint(1, 12, n_samples),
    "skill_count": np.random.randint(1, 25, n_samples),
    "habit_strength": np.random.randint(5, 200, n_samples),
    "openness": np.random.randint(40, 100, n_samples),
    "conscientiousness": np.random.randint(40, 100, n_samples),
    "extraversion": np.random.randint(30, 100, n_samples),
    "agreeableness": np.random.randint(30, 100, n_samples),
    "emotional_stability": np.random.randint(30, 100, n_samples),
    "field": np.random.choice(["Technology", "Business", "Research", "Creative"], n_samples)
})

# Target (Simulated Realistic Formula)
data["success_probability"] = (
    data["daily_hours"] * 4 +
    data["skill_count"] * 2 +
    data["conscientiousness"] * 0.4 +
    data["emotional_stability"] * 0.3 +
    np.random.normal(0, 10, n_samples)
).clip(0, 100)

# ML Pipeline

X = data.drop("success_probability", axis=1)
y = data["success_probability"]

categorical_features = ["field"]
numeric_features = X.columns.drop("field")

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(), categorical_features)
], remainder="passthrough")

model = RandomForestRegressor(n_estimators=200)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X, y)

# Save model
with open("career_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model trained and saved as career_model.pkl")
