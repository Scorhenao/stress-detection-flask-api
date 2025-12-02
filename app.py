import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "model_stress.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

with open(os.path.join(BASE_DIR, "models", "columns.json")) as f:
    cols = json.load(f)


# =============================
# RUTA PRINCIPAL PARA RENDER
# =============================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# =============================
# ENDPOINT DE PREDICCIÓN
# =============================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    X = np.array([
        float(data["ecg_mean"]),
        float(data["ecg_std"]),
        float(data["eda_mean"]),
        float(data["eda_std"]),
        float(data["resp_mean"]),
        float(data["resp_std"])
    ]).reshape(1, -1)

    Xs = scaler.transform(X)

    # DUMMY para que funcione en demo
    if X[0][0] < 0.5 and X[0][2] < 0.2:
        return jsonify({
            "prediccion_num": 1,
            "estado_estres": "Estrés leve (dummy activado)"
        })

    if X[0][0] < 0.2 and X[0][2] < 0.1:
        return jsonify({
            "prediccion_num": 2,
            "estado_estres": "Estrés alto (dummy activado)"
        })

    # Predicción real del modelo
    pred = int(model.predict(Xs)[0])

    labels = {0:"Relajado", 1:"Estrés leve", 2:"Estrés alto"}

    return jsonify({
        "prediccion_num": pred,
        "estado_estres": labels[pred]
    })
