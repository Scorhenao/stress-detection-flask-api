from flask import Flask, request, jsonify
import joblib
import numpy as np
import json
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Cargar modelo y scaler
model = joblib.load("model_stress.pkl")
scaler = joblib.load("scaler.pkl")

# Cargar columnas usadas
columns = json.load(open("columns.json", "r"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Convertir a arreglo
        input_arr = np.array([data[col] for col in columns]).reshape(1, -1)

        # Escalar
        input_scaled = scaler.transform(input_arr)

        # Predecir
        pred = model.predict(input_scaled)[0]

        labels = {
            0: "Relajado",
            1: "Estrés leve",
            2: "Estrés alto"
        }

        return jsonify({
            "prediccion_num": int(pred),
            "estado_estres": labels[int(pred)]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
