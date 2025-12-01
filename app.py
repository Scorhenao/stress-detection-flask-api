import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    stress_label = None

    if request.method == "POST":
        try:
            features = [
                float(request.form["ecg_mean"]),
                float(request.form["ecg_std"]),
                float(request.form["eda_mean"]),
                float(request.form["eda_std"]),
                float(request.form["resp_mean"]),
                float(request.form["resp_std"])
            ]

            # Convertir a array 2D
            X_input = np.array(features).reshape(1, -1)

            # ESCALAR ANTES DE PREDECIR
            X_scaled = scaler.transform(X_input)

            # Predicción
            pred = model.predict(X_scaled)[0]
            prediction = int(pred)

            labels = {0: "Relajado", 1: "Estrés leve", 2: "Estrés alto"}
            stress_label = labels[pred]

        except:
            prediction = "Error"
            stress_label = "Entrada inválida"

    return render_template("index.html", prediction=prediction, stress_label=stress_label)


if __name__ == "__main__":
    app.run(debug=True)
