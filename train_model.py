"""
Run this script ONCE to train and save the model before deploying.
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"          # folder containing the 5 CSV files
MODEL_DIR  = "model"
FILES = {
    "HCLTECH": "HCLTECH.csv",
    "INFY":    "INFOSYS.csv",
    "TCS":     "TCS.csv",
    "TECHM":   "TECHM.csv",
    "WIPRO":   "WIPRO.csv",
}
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load & clean data ─────────────────────────────────────────────────────────
def clean_number(s):
    if pd.isna(s):
        return np.nan
    return float(str(s).replace(",", "").strip())

dfs = []
for symbol, fname in FILES.items():
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Symbol"] = symbol          # normalise symbol name
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Clean column names
df.rename(columns={
    "Open Price":             "Open",
    "High Price":             "High",
    "Low Price":              "Low",
    "Close Price":            "Close",
    "Total Traded Quantity":  "Volume",
    "Date":                   "Date",
}, inplace=True)

for col in ["Open", "High", "Low", "Close", "Volume"]:
    df[col] = df[col].apply(clean_number)

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"], inplace=True)

df["Day"]   = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Year"]  = df["Date"].dt.year

# ── Encode symbol ─────────────────────────────────────────────────────────────
stock_encoder = LabelEncoder()
df["Symbol_Encoded"] = stock_encoder.fit_transform(df["Symbol"])

# ── Target: next-day close (shift -1 within each symbol group) ─────────────
df.sort_values(["Symbol", "Date"], inplace=True)
df["Next_Close"] = df.groupby("Symbol")["Close"].shift(-1)
df.dropna(subset=["Next_Close"], inplace=True)

# ── Features ──────────────────────────────────────────────────────────────────
FEATURES = ["Open", "High", "Low", "Close", "Volume",
            "Day", "Month", "Year", "Symbol_Encoded"]

X = df[FEATURES]
y = df["Next_Close"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Train ─────────────────────────────────────────────────────────────────────
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train_s, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = model.predict(X_test_s)
print(f"MAE : {mean_absolute_error(y_test, preds):.4f}")
print(f"R²  : {r2_score(y_test, preds):.4f}")

# ── Save artefacts ────────────────────────────────────────────────────────────
joblib.dump(model,         os.path.join(MODEL_DIR, "stock_model.pkl"))
joblib.dump(scaler,        os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(stock_encoder, os.path.join(MODEL_DIR, "stock_encoder.pkl"))

print("✅  Model artefacts saved to", MODEL_DIR)
print("Stocks encoded:", dict(zip(stock_encoder.classes_,
                                   stock_encoder.transform(stock_encoder.classes_))))
