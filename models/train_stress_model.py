# models/train_stress_model.py
import os
import numpy as np
import pandas as pd
import wfdb
import json
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib


# ==============================
#   CONFIGURACIÓN
# ==============================

DATA_DIR = "./Data"  # Carpeta con driveXX.dat, driveXX.hea, RECORDS
np.random.seed(42)


# ==============================
#   FUNCIONES AUXILIARES
# ==============================

def find_channel(sig_names, keywords):
    sig_names = [s.lower() for s in sig_names]
    for i, name in enumerate(sig_names):
        if any(kw in name for kw in keywords):
            return i
    return None


def extract_features(record_name, window_seconds=10):
    path = os.path.join(DATA_DIR, record_name)
    rec = wfdb.rdrecord(path)

    signals = rec.p_signal
    sig_names = rec.sig_name
    fs = rec.fs

    n_samples = signals.shape[0]
    win = int(fs * window_seconds)

    idx_ecg  = find_channel(sig_names, ["ecg"])
    idx_eda  = find_channel(sig_names, ["eda", "gsr"])
    idx_resp = find_channel(sig_names, ["resp", "rsp"])

    seg_len = n_samples // 3
    segments = [
        (0, 0, seg_len),              # Relax
        (1, seg_len, seg_len * 2),    # Estrés leve
        (2, seg_len * 2, n_samples)   # Estrés alto
    ]

    rows = []
    for label, start, end in segments:
        for ws in range(start, end - win, win):
            we = ws + win

            row = {
                "ecg_mean":  np.mean(signals[ws:we, idx_ecg])  if idx_ecg  is not None else 0,
                "ecg_std":   np.std(signals[ws:we, idx_ecg])   if idx_ecg  is not None else 0,
                "eda_mean":  np.mean(signals[ws:we, idx_eda])  if idx_eda  is not None else 0,
                "eda_std":   np.std(signals[ws:we, idx_eda])   if idx_eda  is not None else 0,
                "resp_mean": np.mean(signals[ws:we, idx_resp]) if idx_resp is not None else 0,
                "resp_std":  np.std(signals[ws:we, idx_resp])  if idx_resp is not None else 0,
                "stress_level": label
            }

            rows.append(row)

    return rows


# ==============================
#   CARGAR DATASET
# ==============================

records = open(os.path.join(DATA_DIR, "RECORDS")).read().strip().split("\n")

all_rows = []
for rec in records:
    all_rows.extend(extract_features(rec))

df = pd.DataFrame(all_rows)

numeric_cols = ["ecg_mean","ecg_std","eda_mean","eda_std","resp_mean","resp_std"]
X = df[numeric_cols]
y = df["stress_level"]


# ==============================
#  BALANCEO + ESCALADO
# ==============================

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)


# ==============================
#  ENTRENAR MODELO FINAL
# ==============================

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_scaled, y_res)


# ==============================
#  GUARDAR MODELO + SCALER
# ==============================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model_stress.pkl")
joblib.dump(scaler, "models/scaler.pkl")
json.dump(numeric_cols, open("models/columns.json", "w"))

print("✔ Modelo guardado correctamente")
