import numpy as np


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    pass


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    k = np.max(targets) + 1
    conf_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            conf_matrix[i][j] = np.count_nonzero((predictions == j) & (targets == i))
    return conf_matrix


def accuracy(conf_matrix: np.ndarray) -> float:
    return np.trace(conf_matrix) / np.sum(conf_matrix)
