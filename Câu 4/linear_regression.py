import numpy as np

class LinearRegression:
    def __init__(self):
        self._W = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.column_stack((np.ones(len(X)), X))
        self._W = np.linalg.lstsq(X, y, rcond=None)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.column_stack((np.ones(len(X)), X))
        return np.dot(X, self._W)

    def log_invert_transform(self, y: np.ndarray) -> np.ndarray:
        return np.expm1(y)

    def evaluate(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        y = self.log_invert_transform(y)
        y_hat = self.log_invert_transform(y_hat)
        return float(np.sqrt(np.mean(np.square(y - y_hat))))
