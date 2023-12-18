import numpy as np


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    return np.sum(np.power(np.subtract(targets, predictions), 2)) / predictions.size
