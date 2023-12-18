import numpy as np

import utils.metrics as metrics
from config.adaboost_config import cfg as adaboost_cfg
from datasets.dataset_titanic import Titanic
from models.adaboost import Adaboost


def experiment():
    titanic = Titanic(adaboost_cfg)
    adaboost_model = Adaboost(70)
    adaboost_model.train(titanic()['train_input'], titanic()['train_target'])
    predictions = adaboost_model.get_predictions(titanic()['test_input'])
    confusion_matrix = metrics.confusion_matrix(predictions, titanic()['test_target'])
    accuracy = metrics.accuracy(confusion_matrix)
    precision = metrics.precision(confusion_matrix)[0]
    recall = metrics.recall(confusion_matrix)[0]
    f1_score = metrics.f1_score(confusion_matrix)[0]
    print('Confusion matrix on test set:')
    print(confusion_matrix)
    print(f'Accuracy on test set: {accuracy}')
    print(f'Precision on test set: {precision}')
    print(f'Recall on test set: {recall}')
    print(f'F1_score on test set: {f1_score}')


if __name__ == '__main__':
    experiment()

    """
    t = np.array([-1, 0, 1])
    x = np.array([1., -1., 1., 1., 1., 1., -1., -1])
    y = np.array([10., 30., 10., 10., 10., 10., 30., 30.])
    unique_classes, inverse = np.unique(x, return_inverse=True)
    z = np.zeros(3)

    indices = t.argsort()
    indices = indices[np.searchsorted(t, unique_classes, sorter=indices)]

    z[indices] = np.bincount(inverse, weights=y)
    n = np.sum(y)
    print(unique_classes)
    print(z)
    print(np.sum(x) + np.sum(y))
    """
