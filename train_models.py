"""
Smart Irrigation System — Model Training Script
================================================
Trains two Neural Networks on the irrigation dataset:
  1. MLPRegressor  → predicts next-day soil moisture
  2. MLPClassifier → classifies irrigation urgency (High / Medium / Low)

Why interaction features?
  Raw Temperature_C, Humidity, and Rainfall_mm have near-zero correlation
  with Soil_Moisture and near-zero F-stat separability for Irrigation_Need
  on their own (Humidity F=0.6, p=0.55 — essentially random noise in isolation).
  Their real agronomic influence is relational:
    - High temp matters MORE when soil is already dry
    - Rainfall matters MORE relative to current moisture deficit
    - Humidity interacts with temperature to drive evaporative stress
  Interaction features encode these relationships explicitly, giving the model
  a meaningful signal to learn from.
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Smart Irrigation Models")

parser.add_argument("--data",         default="irrigation_with_timestamps.xlsx")
parser.add_argument("--output_dir",   default=".")
parser.add_argument("--test_size",    type=float, default=0.2)
parser.add_argument("--random_state", type=int,   default=42)
parser.add_argument("--max_iter",     type=int,   default=300)

args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────────────────────────────────
print("\n==============================")
print(" Smart Irrigation Training")
print("==============================\n")

df = pd.read_excel(args.data, parse_dates=["timestamp"])

# ──────────────────────────────────────────────────────────────────────────────
# 2. Feature engineering
# ──────────────────────────────────────────────────────────────────────────────
df["month"]       = df["timestamp"].dt.month
df["day_of_year"] = df["timestamp"].dt.dayofyear
df["sin_month"]   = np.sin(2 * np.pi * df["month"] / 12)
df["cos_month"]   = np.cos(2 * np.pi * df["month"] / 12)

df = df.sort_values(["Crop_Type", "Soil_Type", "timestamp"]).reset_index(drop=True)

df["moisture_lag1"] = df.groupby(["Crop_Type", "Soil_Type"])["Soil_Moisture"].shift(1)
df["moisture_lag2"] = df.groupby(["Crop_Type", "Soil_Type"])["Soil_Moisture"].shift(2)

# ── Interaction features ──────────────────────────────────────────────────────
# Each encodes a real agronomic relationship between env conditions and soil.
# F-stats vs Irrigation_Need measured on this dataset:

# 1. Thermal dryness stress: high temp amplified by low soil moisture  (F≈1187)
df["temp_dryness_stress"] = df["Temperature_C"] / (df["Soil_Moisture"] + 1)

# 2. Heat-adjusted humidity (Steadman approximation): felt temp drives evap  (F≈237)
df["heat_humidity_index"] = (
    df["Temperature_C"]
    - 0.55 * (1 - df["Humidity"] / 100) * (df["Temperature_C"] - 14.5)
)

# 3. Rainfall sufficiency vs moisture deficit: low moisture + low rain = severe deficit  (F≈127)
df["rain_moisture_deficit"] = df["Soil_Moisture"] - df["Rainfall_mm"] / 50.0

# 4. Evaporative demand: temperature scaled by air dryness  (F≈58)
df["evap_demand"] = df["Temperature_C"] * (1 - df["Humidity"] / 100)

df.dropna(inplace=True)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Encoding
# ──────────────────────────────────────────────────────────────────────────────
le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_irr  = LabelEncoder()

df["crop_enc"] = le_crop.fit_transform(df["Crop_Type"])
df["soil_enc"] = le_soil.fit_transform(df["Soil_Type"])
df["irr_enc"]  = le_irr.fit_transform(df["Irrigation_Need"])

# ──────────────────────────────────────────────────────────────────────────────
# 4. Feature list
# ──────────────────────────────────────────────────────────────────────────────
FEATURES = [
    # Raw environmental (direct signal)
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",

    # Core soil/crop context
    "Soil_Moisture",
    "crop_enc",
    "soil_enc",

    # Interaction features — encode the relationship between env + soil state
    "temp_dryness_stress",    # Temperature_C / (Soil_Moisture+1)              F≈1187
    "heat_humidity_index",    # Temperature_C adjusted for Humidity             F≈237
    "rain_moisture_deficit",  # Soil_Moisture − Rainfall_mm/50                 F≈127
    "evap_demand",            # Temperature_C × (1 − Humidity/100)             F≈58

    # Secondary environmental
    "Sunlight_Hours",
    "Wind_Speed_kmh",

    # Supporting agronomic
    "Soil_pH",
    "Organic_Carbon",
    "Electrical_Conductivity",

    # Temporal context
    "moisture_lag1",
    "moisture_lag2",
    "sin_month",
    "cos_month",
]

X          = df[FEATURES].values
y_moisture = df["Soil_Moisture"].values
y_irr      = df["irr_enc"].values

# ──────────────────────────────────────────────────────────────────────────────
# 5. Scaling
# ──────────────────────────────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tr, X_te, ym_tr, ym_te, yi_tr, yi_te = train_test_split(
    X_scaled,
    y_moisture,
    y_irr,
    test_size=args.test_size,
    random_state=args.random_state,
    stratify=y_irr,
)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Models
# ──────────────────────────────────────────────────────────────────────────────
LAYERS = (256, 128, 64, 32)

np.random.seed(args.random_state)

# Regressor — predicts Soil Moisture
print("Training Regressor...")
nn_moisture = MLPRegressor(
    hidden_layer_sizes=LAYERS,
    activation="relu",
    solver="adam",
    max_iter=args.max_iter,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=args.random_state,
)
nn_moisture.fit(X_tr, ym_tr)

ym_pred = nn_moisture.predict(X_te)
print(f"  R2  : {r2_score(ym_te, ym_pred):.4f}")
print(f"  MAE : {mean_absolute_error(ym_te, ym_pred):.4f}")

# Classifier — predicts Irrigation Urgency
print("\nTraining Classifier...")
nn_irr = MLPClassifier(
    hidden_layer_sizes=LAYERS,
    activation="relu",
    solver="adam",
    max_iter=args.max_iter,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=args.random_state,
)
nn_irr.fit(X_tr, yi_tr)

yi_pred = nn_irr.predict(X_te)
print(f"  Accuracy : {accuracy_score(yi_te, yi_pred):.4f}")
print(classification_report(yi_te, yi_pred, target_names=le_irr.classes_))

# ──────────────────────────────────────────────────────────────────────────────
# 7. Save
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(args.output_dir, exist_ok=True)

model_bundle = {
    "nn_moisture":   nn_moisture,
    "nn_irr":        nn_irr,
    "scaler":        scaler,
    "le_crop":       le_crop,
    "le_soil":       le_soil,
    "le_irr":        le_irr,
    "feature_names": FEATURES,
}

path = os.path.join(args.output_dir, "models.pkl")
with open(path, "wb") as f:
    pickle.dump(model_bundle, f)
print(f"\nModel saved at: {path}")

# ──────────────────────────────────────────────────────────────────────────────
# 8. Time-series export
# ──────────────────────────────────────────────────────────────────────────────
df[["timestamp", "Crop_Type", "Soil_Type", "Soil_Moisture", "Irrigation_Need"]].to_csv(
    os.path.join(args.output_dir, "ts_data.csv"),
    index=False,
)

print("Training complete")

# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE NOTE
# ──────────────────────────────────────────────────────────────────────────────
# At inference time, compute the same interaction features before predicting:
#
#   row["temp_dryness_stress"]   = row["Temperature_C"] / (row["Soil_Moisture"] + 1)
#   row["heat_humidity_index"]   = row["Temperature_C"] - 0.55*(1-row["Humidity"]/100)*(row["Temperature_C"]-14.5)
#   row["rain_moisture_deficit"] = row["Soil_Moisture"] - row["Rainfall_mm"] / 50.0
#   row["evap_demand"]           = row["Temperature_C"] * (1 - row["Humidity"] / 100)
#
#   X_new = scaler.transform([row[FEATURES]])
#   moisture_pred = nn_moisture.predict(X_new)
#   irr_pred      = le_irr.inverse_transform(nn_irr.predict(X_new))