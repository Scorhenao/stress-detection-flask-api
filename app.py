import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load("model_stress(1).pkl")

scaler = joblib.load("scaler(2).pkl") 


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = [
            float(data["ecg_mean"]),
            float(data["ecg_std"]),
            float(data["eda_mean"]),
            float(data["eda_std"]),
            float(data["resp_mean"]),
            float(data["resp_std"])
        ]

        X_input = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X_input)
        pred = int(model.predict(X_scaled)[0])

        labels = {0: "Relajado", 1: "Estrés leve", 2: "Estrés alto"}

        return {
            "prediccion_num": pred,
            "estado_estres": labels[pred]
        }

    except Exception as e:
        return {"error": str(e)}, 400


if __name__ == "__main__":
    app.run(debug=True)
