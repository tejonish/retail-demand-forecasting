import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Retail Demand Forecasting",
    layout="wide"
)

# --------------------------------------------------
# Load Model & Feature Columns
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models/xgb_model.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "models/feature_columns.pkl"))

# --------------------------------------------------
# Load Cleaned Data
# --------------------------------------------------
data = pd.read_csv(os.path.join(BASE_DIR, "data/processed/cleaned_data.csv"))
data["Date"] = pd.to_datetime(data["Date"])

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📊 Retail Demand Forecasting Dashboard")
st.write("Recursive 7-day demand forecasting using XGBoost.")

store_id = st.selectbox("Select Store", sorted(data["Store"].unique()))
forecast_date = st.date_input("Select Forecast Start Date")

# --------------------------------------------------
# Forecast Logic
# --------------------------------------------------
if st.button("Generate 7-Day Forecast"):

    forecast_date = pd.to_datetime(forecast_date)

    store_data = data[data["Store"] == store_id].sort_values("Date").copy()

    # Check valid date
    if forecast_date < store_data["Date"].min():
        st.error("Selected date is before available historical data.")
        st.stop()

    # Use only historical data up to selected date
    future_data = store_data[store_data["Date"] <= forecast_date].copy()

    if len(future_data) < 30:
        st.error("Not enough historical data before selected date (need at least 30 days).")
        st.stop()

    forecast_days = 7
    predictions = []

    for _ in range(forecast_days):

        last_row = future_data.iloc[-1:].copy()
        next_date = future_data["Date"].max() + pd.Timedelta(days=1)

        new_row = last_row.copy()
        new_row["Date"] = next_date

        # Update calendar features
        new_row["Year"] = next_date.year
        new_row["Month"] = next_date.month
        new_row["Day"] = next_date.day
        new_row["DayOfWeek"] = next_date.weekday()

        # Update lag features
        new_row["Lag_1"] = future_data["Sales"].iloc[-1]
        new_row["Lag_7"] = future_data["Sales"].iloc[-7]

        # Update rolling features
        new_row["Rolling_Mean_7"] = future_data["Sales"].iloc[-7:].mean()
        new_row["Rolling_Mean_14"] = future_data["Sales"].iloc[-14:].mean()
        new_row["Rolling_Mean_30"] = future_data["Sales"].iloc[-30:].mean()

        # Prepare model input
        input_df = pd.get_dummies(new_row)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Predict
        pred = model.predict(input_df)[0]
        new_row["Sales"] = pred

        predictions.append((next_date, pred))

        # Append prediction to dataset
        future_data = pd.concat([future_data, new_row], ignore_index=True)

    forecast_df = pd.DataFrame(predictions, columns=["Date", "Predicted Sales"])

    # Convert to date-only (remove time)
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.date

    # --------------------------------------------------
    # Display 7-Day Forecast
    # --------------------------------------------------
    st.subheader("📈 7-Day Forecast")
    st.line_chart(forecast_df.set_index("Date"))

    # --------------------------------------------------
    # Historical vs Forecast Comparison
    # --------------------------------------------------
    st.subheader("📊 Historical vs Forecast")

    recent_history = (
        store_data[store_data["Date"] <= forecast_date]
        .tail(14)[["Date", "Sales"]]
    )

    # Convert to date-only
    recent_history["Date"] = pd.to_datetime(recent_history["Date"]).dt.date
    recent_history = recent_history.set_index("Date")

    forecast_plot = forecast_df.rename(
        columns={"Predicted Sales": "Sales"}
    ).set_index("Date")

    combined = pd.concat([recent_history, forecast_plot])

    st.line_chart(combined)

    st.caption(f"Forecast begins after {forecast_date.date()}")
