import os

try:
    import pandas as pd  # type: ignore[reportMissingModuleSource]
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: pandas. Install it with: python -m pip install pandas scikit-learn joblib"
    ) from exc

import joblib  # type: ignore[reportMissingImports]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------
# 1. Load dataset
# -----------------------------------

DATA_PATH = "data/heart.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# -----------------------------------
# 2. Separate features and target
# -----------------------------------

TARGET = "target"

X = df.drop(TARGET, axis=1)
y = df[TARGET]


# -----------------------------------
# 3. Train-test split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# -----------------------------------
# 4. Create ML pipeline
# -----------------------------------

pipeline = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


# -----------------------------------
# 5. Train model
# -----------------------------------

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)


# -----------------------------------
# 6. Make predictions
# -----------------------------------

y_pred = pipeline.predict(X_test)


# -----------------------------------
# 7. Evaluate model
# -----------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n================================")
print("MODEL RESULTS")
print("================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# -----------------------------------
# 8. Create model folder
# -----------------------------------

os.makedirs(
    "model",
    exist_ok=True
)


# -----------------------------------
# 9. Save model
# -----------------------------------

MODEL_PATH = "model/heart_model.pkl"

joblib.dump(
    pipeline,
    MODEL_PATH
)


print("\n================================")
print("SUCCESS")
print("================================")

print(
    f"Model saved successfully!"
)

print(
    f"Location: {MODEL_PATH}"
)