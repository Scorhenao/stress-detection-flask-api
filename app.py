import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

# ============================
# CARGA DEL MODELO Y ESCALER
# ============================
MODEL_DIR = "models"

model = joblib.load(os.path.join(MODEL_DIR, "model_stress.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# Cargar columnas originales
columns = json.load(open(os.path.join(MODEL_DIR, "columns.json")))


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convertir a lista en el mismo orden del entrenamiento
        features = np.array([
            float(data["ecg_mean"]),
            float(data["ecg_std"]),
            float(data["eda_mean"]),
            float(data["eda_std"]),
            float(data["resp_mean"]),
            float(data["resp_std"])
        ]).reshape(1, -1)

        # Escalar
        features_scaled = scaler.transform(features)

        # Predecir
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
