import pandas as pd

FEATURES = ["hour", "day_of_week", "month", "is_weekend",
            "temp", "rain_1h", "clouds_all"]
TARGET = "traffic_volume"

def load_and_engineer(path):
    df = pd.read_csv(path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df
