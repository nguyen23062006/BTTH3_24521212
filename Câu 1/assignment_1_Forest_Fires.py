from pathlib import Path
import numpy as np
from linear_regression import LinearRegression


def convert_month2int(month: str) -> int:
    months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,"sep": 9, "oct": 10, "nov": 11, "dec": 12,}
    return months[month]

def convert_date2int(date: str) -> int:
    days = {"sun": 1, "mon": 2, "tue": 3, "wed": 4,"thu": 5, "fri": 6, "sat": 7,}
    return days[date]

def preprocess_data(data: np.ndarray) -> np.ndarray:
    data = data.copy()
    data[:, 2] = [convert_month2int(month) for month in data[:, 2]]
    data[:, 3] = [convert_date2int(day) for day in data[:, 3]]
    data = data.astype(float)
    data[:, -1] = np.log1p(data[:, -1])
    return data

def organize_data(data: np.ndarray):
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent/"data"/"forestfires.csv"
    data = np.genfromtxt(data_path, delimiter=",", dtype=str, skip_header=1)
    data = preprocess_data(data)
    data = data[np.random.default_rng(42).permutation(len(data))]

    X_train, y_train = organize_data(data[:414])
    X_test, y_test = organize_data(data[414:])

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_predicted = lr_model.predict(X_test)
    rmse = lr_model.evaluate(y_test, y_predicted)

    print("The root mean squared error is:", rmse)
