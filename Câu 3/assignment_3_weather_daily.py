import csv
from pathlib import Path
import numpy as np
from linear_regression import LinearRegression


def load_weather(path):
    feature_names = ["temperature_2m_max","temperature_2m_min","temperature_2m_mean","sunshine_duration","wind_speed_10m_max",]
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    X = np.array([[float(row[col]) for col in feature_names] for row in rows])
    y = np.log1p([float(row["precipitation_sum"]) for row in rows])
    dates = np.array([row["time"] for row in rows])
    return X, y, dates, feature_names

def select_features(X_train, threshold=0.8):
    correlation = np.abs(np.corrcoef(X_train, rowvar=False))
    selected = []
    for column in range(X_train.shape[1]):
        if all(correlation[column, previous] < threshold for previous in selected):
            selected.append(column)
    return selected

if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent / "data" / "weather_daily.csv"
    X, y, dates, feature_names = load_weather(data_path)

    train_mask = (dates >= "2021-01-01") & (dates < "2025-01-01")
    test_mask = (dates >= "2025-01-01") & (dates < "2026-01-01")
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    selected = select_features(X_train)
    removed = [feature_names[i] for i in range(len(feature_names)) if i not in selected]
    print("Removed columns:", ", ".join(removed))

    for name, train, test in [("Original data", X_train, X_test),("Preprocessed data", X_train[:, selected], X_test[:, selected]),]:
        lr_model = LinearRegression()
        lr_model.fit(train, y_train)
        y_predicted = lr_model.predict(test)
        rmse = lr_model.evaluate(y_test, y_predicted)
        print(f"RMSE({name}):", rmse)
