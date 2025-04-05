# 📊 Streamlit Dashboard for Amazon Price Forecasting

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Amazon Price Forecasting Tool", layout="centered")
st.title("📈 Amazon Product Price Forecasting")

# --- Sidebar config ---
st.sidebar.header("⚙️ Configuration")
forecast_days = st.sidebar.slider("Forecast days", min_value=7, max_value=90, value=30)
apply_valentine_override = st.sidebar.checkbox("Force dip to €160 during Valentine's sale", value=True)

# --- File upload ---
uploaded_file = st.file_uploader("Upload your product price CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Price'] = df['Price'].astype(str).str.replace(',', '.').astype(float)
    df = df.dropna(subset=['Date', 'Price']).sort_values('Date').reset_index(drop=True)

    # --- Feature Engineering ---
    df['price_lag_1'] = df['Price'].shift(1)
    df['price_lag_3'] = df['Price'].shift(3)
    df['price_lag_7'] = df['Price'].shift(7)
    df['rolling_avg_3'] = df['Price'].rolling(3).mean()
    df['rolling_avg_7'] = df['Price'].rolling(7).mean()
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['Date'].dt.month

    # --- Event Flags ---
    valentines = pd.date_range("2025-01-25", "2025-02-14")
    df['is_valentines_sale_season'] = df['Date'].isin(valentines).astype(int)

    features = [
        'price_lag_1', 'price_lag_3', 'price_lag_7',
        'rolling_avg_3', 'rolling_avg_7',
        'day_of_week', 'is_weekend', 'month', 'is_valentines_sale_season']

    df_model = df.dropna()
    X = df_model[features]
    y = df_model['Price']

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)

    # --- Forecasting ---
    last_known = df_model.tail(7).copy()
    predictions = []
    current_date = df['Date'].max() + pd.Timedelta(days=1)
    anchor_prices = sorted(df['Price'].round(-1).unique())[::-1]  # Use historical rounded anchors

    for _ in range(forecast_days):
        dow = current_date.dayofweek
        month = current_date.month
        is_weekend = int(dow in [5, 6])
        is_valentines = int(current_date in valentines)

        row = pd.DataFrame([{
            'price_lag_1': last_known['Price'].iloc[-1],
            'price_lag_3': last_known['Price'].iloc[-3],
            'price_lag_7': last_known['Price'].iloc[-7],
            'rolling_avg_3': last_known['Price'].iloc[-3:].mean(),
            'rolling_avg_7': last_known['Price'].iloc[-7:].mean(),
            'day_of_week': dow,
            'is_weekend': is_weekend,
            'month': month,
            'is_valentines_sale_season': is_valentines
        }])

        pred_price = model.predict(row)[0]
        rounded = min(anchor_prices, key=lambda x: abs(x - pred_price))

        if apply_valentine_override and is_valentines and pred_price < 195:
            final = 160
        else:
            final = rounded

        predictions.append({
            'Date': current_date,
            'Predicted Price': pred_price,
            'Rounded Prediction': rounded,
            'Final Adjusted Prediction': final
        })

        last_known = pd.concat([last_known, pd.DataFrame([{'Date': current_date, 'Price': pred_price}])], ignore_index=True)
        current_date += pd.Timedelta(days=1)

    forecast_df = pd.DataFrame(predictions)

    # --- Display Forecast ---
    st.subheader("📉 Forecasted Prices")
    st.line_chart(forecast_df.set_index('Date')[['Final Adjusted Prediction']])

    st.dataframe(forecast_df, use_container_width=True)

    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast as CSV", data=csv, file_name="forecast_output.csv", mime="text/csv")

else:
    st.info("👈 Upload a product price CSV to get started.")
