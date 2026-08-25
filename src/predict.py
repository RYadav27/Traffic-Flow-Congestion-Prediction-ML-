import joblib
import pandas as pd
from preprocess import FEATURES

def predict_traffic(model_path, date_time, temp, rain_1h, clouds_all):
    dt = pd.to_datetime(date_time)
    row = pd.DataFrame([{
        "hour": dt.hour,
        "day_of_week": dt.dayofweek,
        "month": dt.month,
        "is_weekend": int(dt.dayofweek >= 5),
        "temp": temp,
        "rain_1h": rain_1h,
        "clouds_all": clouds_all
    }])
    model = joblib.load(model_path)
    return float(model.predict(row[FEATURES])[0])
