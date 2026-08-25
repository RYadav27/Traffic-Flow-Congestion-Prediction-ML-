# Traffic Flow / Congestion Prediction

A transportation-engineering machine-learning project that predicts hourly traffic
volume from time and weather features using a Random Forest Regressor.

## Project structure

```text
traffic-flow-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── data/
│   └── metro_traffic.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── models/
│   └── model.pkl
└── notebooks/
```

## Important note about the included Excel/CSV-style data

The included dataset is a **synthetic sample dataset** created for this portfolio
project so that the repository runs immediately. It is not the official UCI
Metro Interstate Traffic Volume dataset and should not be described as real UCI
data.

For a real-data version, replace `data/metro_traffic.csv` with the UCI Metro
Interstate Traffic Volume dataset and map its columns to the features used here.

## Features

- Hour
- Day of week
- Month
- Weekend indicator
- Temperature
- Rain in previous hour
- Cloud coverage

## Model

Random Forest Regression.

## How to run

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

## Output

The training script reports MAE, RMSE and R² on the held-out test set and saves
the trained model as `models/model.pkl`.

## Resume project description

Built a Streamlit traffic-flow prediction dashboard using Random Forest regression,
engineering temporal and weather features to estimate hourly traffic volume.
