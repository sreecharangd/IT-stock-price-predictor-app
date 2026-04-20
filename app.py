"""
Stock Price Prediction App — IT Sector (India)
Streamlit front-end with a built-in training fallback so the app works
on Streamlit Cloud without pre-built model files.
"""

import os, io, re
import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IT Stock Price Predictor",
    page_icon="📈",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Training model on first run …")
def load_or_train_model():
    """Load saved model artefacts or train from CSV data if they don't exist."""
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split

    MODEL_DIR = "model"
    MODEL_PATH   = os.path.join(MODEL_DIR, "stock_model.pkl")
    SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
    ENCODER_PATH = os.path.join(MODEL_DIR, "stock_encoder.pkl")

    # ── Try loading pre-trained artefacts ────────────────────────────────────
    if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]):
        model         = joblib.load(MODEL_PATH)
        scaler        = joblib.load(SCALER_PATH)
        stock_encoder = joblib.load(ENCODER_PATH)
        return model, scaler, stock_encoder

    # ── Otherwise train on the fly ────────────────────────────────────────────
    DATA_DIR = "data"
    FILES = {
        "HCLTECH": "HCLTECH.csv",
        "INFY":    "INFOSYS.csv",
        "TCS":     "TCS.csv",
        "TECHM":   "TECHM.csv",
        "WIPRO":   "WIPRO.csv",
    }

    def clean_number(s):
        if pd.isna(s):
            return np.nan
        return float(re.sub(r"[^\d.\-]", "", str(s)) or "nan")

    dfs = []
    for symbol, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            st.error(f"Missing data file: {path}")
            st.stop()
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df["Symbol"] = symbol
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.rename(columns={
        "Open Price":            "Open",
        "High Price":            "High",
        "Low Price":             "Low",
        "Close Price":           "Close",
        "Total Traded Quantity": "Volume",
        "Date":                  "Date",
    }, inplace=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].apply(clean_number)

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"], inplace=True)
    df["Day"]   = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"]  = df["Date"].dt.year

    stock_encoder = LabelEncoder()
    df["Symbol_Encoded"] = stock_encoder.fit_transform(df["Symbol"])

    df.sort_values(["Symbol", "Date"], inplace=True)
    df["Next_Close"] = df.groupby("Symbol")["Close"].shift(-1)
    df.dropna(subset=["Next_Close"], inplace=True)

    FEATURES = ["Open", "High", "Low", "Close", "Volume",
                "Day", "Month", "Year", "Symbol_Encoded"]
    X = df[FEATURES]
    y = df["Next_Close"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)

    # Persist for subsequent runs
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,         MODEL_PATH)
    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(stock_encoder, ENCODER_PATH)

    return model, scaler, stock_encoder


def load_historical(symbol: str) -> pd.DataFrame:
    """Return cleaned historical DataFrame for the given symbol."""
    fname_map = {
        "HCLTECH": "HCLTECH.csv",
        "INFY":    "INFOSYS.csv",
        "TCS":     "TCS.csv",
        "TECHM":   "TECHM.csv",
        "WIPRO":   "WIPRO.csv",
    }
    path = os.path.join("data", fname_map[symbol])
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df.rename(columns={
        "Open Price":  "Open",
        "High Price":  "High",
        "Low Price":   "Low",
        "Close Price": "Close",
        "Total Traded Quantity": "Volume",
        "Date": "Date",
    }, inplace=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ""), errors="coerce"
        )
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df.dropna(subset=["Date", "Close"], inplace=True)
    df.sort_values("Date", inplace=True)
    return df


# ── Load model ────────────────────────────────────────────────────────────────
model, scaler, stock_encoder = load_or_train_model()

STOCKS = sorted(stock_encoder.classes_.tolist())
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume",
                   "Day", "Month", "Year", "Symbol_Encoded"]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📈 IT Sector Stock Price Predictor")
st.caption("Indian IT stocks · Random Forest model · Predicts next-day closing price")

tab1, tab2 = st.tabs(["🔮 Predict", "📊 Historical Chart"])

# ────────────────────────────── TAB 1 : Predict ──────────────────────────────
with tab1:
    st.warning(
        "⚠️ **Disclaimer**: Predictions are based on historical patterns and are "
        "for educational purposes only. Do not use for actual investment decisions."
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Input Parameters")
        stock = st.selectbox("Select Stock", STOCKS)

        # Pre-fill sensible defaults from the latest row
        hist = load_historical(stock)
        last = hist.iloc[-1]

        open_price  = st.number_input("Open Price (₹)",  value=float(round(last["Open"],  2)), step=0.5, format="%.2f")
        high_price  = st.number_input("High Price (₹)",  value=float(round(last["High"],  2)), step=0.5, format="%.2f")
        low_price   = st.number_input("Low Price (₹)",   value=float(round(last["Low"],   2)), step=0.5, format="%.2f")
        close_price = st.number_input("Close Price (₹)", value=float(round(last["Close"], 2)), step=0.5, format="%.2f")
        volume      = st.number_input("Volume", value=float(int(last["Volume"])), step=1000.0, format="%.0f")

        st.divider()
        st.markdown("**Prediction date**")
        dcol1, dcol2, dcol3 = st.columns(3)
        day   = dcol1.number_input("Day",   value=20, min_value=1, max_value=31)
        month = dcol2.number_input("Month", value=4,  min_value=1, max_value=12)
        year  = dcol3.number_input("Year",  value=2026, min_value=2000, max_value=2100)

        predict_btn = st.button("🚀 Predict Next Close Price", use_container_width=True, type="primary")

    with col2:
        st.subheader("Prediction Result")
        if predict_btn:
            stock_encoded = int(stock_encoder.transform([stock])[0])
            input_df = pd.DataFrame([{
                "Open":           open_price,
                "High":           high_price,
                "Low":            low_price,
                "Close":          close_price,
                "Volume":         volume,
                "Day":            day,
                "Month":          month,
                "Year":           year,
                "Symbol_Encoded": stock_encoded,
            }], columns=FEATURE_COLUMNS)

            input_scaled = scaler.transform(input_df)
            prediction   = model.predict(input_scaled)[0]
            change       = prediction - close_price
            pct_change   = (change / close_price) * 100 if close_price else 0
            direction    = "▲" if change >= 0 else "▼"
            colour       = "green" if change >= 0 else "red"

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
                border-radius: 16px;
                padding: 32px;
                text-align: center;
                border: 1px solid #2e5f9e;
                margin-top: 12px;
            ">
                <div style="color:#aac4e8; font-size:14px; margin-bottom:6px;">
                    {stock} · Predicted Next Close
                </div>
                <div style="color:#ffffff; font-size:48px; font-weight:700; letter-spacing:-1px;">
                    ₹{prediction:,.2f}
                </div>
                <div style="color:{colour}; font-size:20px; margin-top:10px;">
                    {direction} ₹{abs(change):,.2f} ({pct_change:+.2f}%)
                </div>
                <div style="color:#7a9cc0; font-size:12px; margin-top:18px;">
                    vs today's close of ₹{close_price:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Feature importance mini-bar
            st.markdown("#### Feature Importance")
            fi = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
            st.bar_chart(fi)
        else:
            st.info("Fill in the parameters on the left and click **Predict**.")

# ────────────────────────────── TAB 2 : Chart ────────────────────────────────
with tab2:
    chart_stock = st.selectbox("Select Stock for Chart", STOCKS, key="chart_stock")
    hist2 = load_historical(chart_stock)

    st.subheader(f"{chart_stock} — Historical Closing Prices")
    st.line_chart(hist2.set_index("Date")[["Close"]])

    st.subheader("Recent Data (last 20 rows)")
    display_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    st.dataframe(
        hist2[display_cols].tail(20).sort_values("Date", ascending=False).reset_index(drop=True),
        use_container_width=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Streamlit · Random Forest Regressor · "
    "Data: NSE IT sector stocks (HCLTECH, INFY, TCS, TECHM, WIPRO)"
)
