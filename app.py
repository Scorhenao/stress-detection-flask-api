import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

# ============================
# CARGA DEL MODELO Y SCALER
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "model_stress.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler3.pkl"))

with open(os.path.join(BASE_DIR, "models", "columns.json"), "r") as f:
    feature_cols = json.load(f)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Orden correcto según el entrenamiento
        features = np.array([
            float(data["ecg_mean"]),
            float(data["ecg_std"]),
            float(data["eda_mean"]),
            float(data["eda_std"]),
            float(data["resp_mean"]),
            float(data["resp_std"])
        ]).reshape(1, -1)

        # Escalado
        features_scaled = scaler.transform(features)

        # Predicción
        pred = int(model.predict(features_scaled)[0])

        labels = {0: "Relajado", 1: "Estrés leve", 2: "Estrés alto"}

        return jsonify({
            "prediccion_num": pred,
            "estado_estres": labels[pred]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
