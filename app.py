import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys

sys.path.append("src")
from preprocess import FEATURES

st.set_page_config(page_title="Traffic Flow Prediction", page_icon="🚗", layout="wide")
st.title("🚗 Traffic Flow / Congestion Prediction")
st.caption("Random Forest model for hourly traffic-volume prediction")

model_path = Path("models/model.pkl")
data_path = Path("data/metro_traffic.csv")

if not model_path.exists():
    st.warning("Model not found. Run: python src/train.py")
    st.stop()

model = joblib.load(model_path)
data = pd.read_csv(data_path, parse_dates=["date_time"])

left, right = st.columns(2)
with left:
    date_time = st.datetime_input("Date and time", value=pd.Timestamp("2025-06-16 08:00"))
    temp = st.number_input("Temperature (°C)", -30.0, 50.0, 20.0)
    rain = st.number_input("Rain in last hour (mm)", 0.0, 100.0, 0.0)
with right:
    clouds = st.slider("Cloud coverage (%)", 0, 100, 50)
    if st.button("Predict Traffic Volume", type="primary"):
        dt = pd.Timestamp(date_time)
        row = pd.DataFrame([{
            "hour": dt.hour,
            "day_of_week": dt.dayofweek,
            "month": dt.month,
            "is_weekend": int(dt.dayofweek >= 5),
            "temp": temp,
            "rain_1h": rain,
            "clouds_all": clouds
        }])
        prediction = model.predict(row[FEATURES])[0]
        st.metric("Predicted Traffic Volume", f"{prediction:,.0f} vehicles/hour")

st.divider()
st.subheader("Traffic Data Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Records", f"{len(data):,}")
c2.metric("Average Volume", f"{data.traffic_volume.mean():,.0f}")
c3.metric("Maximum Volume", f"{data.traffic_volume.max():,.0f}")

chart = data.set_index("date_time")["traffic_volume"].resample("D").mean()
st.line_chart(chart)
