import os
import json
import joblib
import pandas as pd
from http.server import BaseHTTPRequestHandler


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "heart_model.pkl"
)

model = joblib.load(MODEL_PATH)


class handler(BaseHTTPRequestHandler):

    def do_POST(self):

        try:

            content_length = int(
                self.headers.get("Content-Length", 0)
            )

            body = self.rfile.read(content_length)

            data = json.loads(body.decode("utf-8"))


            required_features = [
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


            missing = [
                feature
                for feature in required_features
                if feature not in data
            ]


            if missing:

                self.send_json(
                    400,
                    {
                        "error":
                        "Missing fields: "
                        + ", ".join(missing)
                    }
                )

                return


            input_data = pd.DataFrame([
                {
                    feature: data[feature]
                    for feature in required_features
                }
            ])


            prediction = model.predict(
                input_data
            )[0]


            probability = model.predict_proba(
                input_data
            )[0]


            positive_probability = (
                probability[1] * 100
            )


            if int(prediction) == 1:

                result = "Heart Disease Detected"

            else:

                result = "No Heart Disease Detected"


            response = {

                "prediction": int(prediction),

                "result": result,

                "probability":
                    round(
                        positive_probability,
                        2
                    )
            }


            self.send_json(
                200,
                response
            )


        except Exception as e:

            self.send_json(
                500,
                {
                    "error": str(e)
                }
            )


    def do_GET(self):

        self.send_json(
            405,
            {
                "error":
                "Only POST requests are allowed"
            }
        )


    def send_json(self, status_code, data):

        response = json.dumps(data).encode(
            "utf-8"
        )


        self.send_response(status_code)

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.send_header(
            "Content-Length",
            str(len(response))
        )

        self.end_headers()

        self.wfile.write(response)
