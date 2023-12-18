import numpy as np


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    unique_targets = np.unique(targets)
    k = len(unique_targets)
    conf_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            conf_matrix[i][j] = np.count_nonzero((predictions == unique_targets[j]) & (targets == unique_targets[i]))
    return conf_matrix


def accuracy(conf_matrix: np.ndarray) -> float:
    return np.trace(conf_matrix) / np.sum(conf_matrix)


def precision(conf_matrix: np.ndarray) -> np.ndarray:
    return np.diagonal(conf_matrix) / np.sum(conf_matrix, axis=0)


def recall(conf_matrix: np.ndarray) -> np.ndarray:
    return np.diagonal(conf_matrix) / np.sum(conf_matrix, axis=1)


def f1_score(conf_matrix: np.ndarray) -> np.ndarray:
    precisions = precision(conf_matrix)
    recalls = recall(conf_matrix)
    return 2 * (precisions * recalls) / (precisions + recalls)
