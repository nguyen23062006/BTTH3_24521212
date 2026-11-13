import csv
from pathlib import Path
import numpy as np
from linear_regression import LinearRegression

def load_weather(path):
    feature_columns = ["temperature_2m_max","temperature_2m_min","temperature_2m_mean","sunshine_duration","wind_speed_10m_max",]
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    X = np.array([[float(row[col]) for col in feature_columns] for row in rows])
    y = np.log1p([float(row["precipitation_sum"]) for row in rows])
    dates = np.array([row["time"] for row in rows])
    return X, y, dates

def standardize_data(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std

if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent/"data"/"weather_daily.csv"
    X, y, dates = load_weather(data_path)

    train_mask = (dates >= "2021-01-01") & (dates < "2025-01-01")
    test_mask = (dates >= "2025-01-01") & (dates < "2026-01-01")
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    X_train_std, X_test_std = standardize_data(X_train, X_test)

    for name, train, test in [("Original data", X_train, X_test),("Standardized data", X_train_std, X_test_std),]:
        lr_model = LinearRegression()
        lr_model.fit(train, y_train)
        y_predicted = lr_model.predict(test)
        rmse = lr_model.evaluate(y_test, y_predicted)
        print(f"RMSE({name}):", rmse)
