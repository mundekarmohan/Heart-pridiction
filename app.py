from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/heart_model.pkl")

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():

    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.get_json()

        input_data = pd.DataFrame([
            {
                feature: data[feature]
                for feature in FEATURES
            }
        ])

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

        positive_probability = probability[1] * 100

        if int(prediction) == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease Detected"

        return jsonify({
            "prediction": int(prediction),
            "result": result,
            "probability": round(
                float(positive_probability), 2
            )
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        debug=True,
        port=5000
    )
