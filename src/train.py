from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from preprocess import load_and_engineer, FEATURES, TARGET

DATA_PATH = Path("data/metro_traffic.csv")
MODEL_PATH = Path("models/model.pkl")

def train():
    df = load_and_engineer(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=250, max_depth=18, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"MAE:  {mae:.2f} vehicles/hour")
    print(f"RMSE: {rmse:.2f} vehicles/hour")
    print(f"R2:   {r2:.4f}")
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
