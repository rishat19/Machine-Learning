import pickle

import numpy as np

import utils.metrics as metrics
from config.logistic_regression_config import cfg as log_reg_cfg
from datasets.digits_dataset import Digits
from models.logistic_regression_model import LogReg
from utils.enums import SetType
from utils.visualisation import Visualisation


def experiment(digits_dataset, log_reg_model: LogReg) -> bytes:
    log_reg_model.train(digits_dataset(SetType.train)['inputs'], digits_dataset(SetType.train)['targets'],
                        digits_dataset(SetType.valid)['inputs'], digits_dataset(SetType.valid)['targets'])
    Visualisation.visualise_metrics(epochs=log_reg_model.data_for_plots['epochs'],
                                    metrics=log_reg_model.data_for_plots['target_function_value_train'],
                                    plot_title='Target function values on training set',
                                    y_title='Target function value')
    Visualisation.visualise_metrics(epochs=log_reg_model.data_for_plots['epochs'],
                                    metrics=log_reg_model.data_for_plots['accuracy_train'],
                                    plot_title='Accuracy values on training set',
                                    y_title='Accuracy')
    Visualisation.visualise_metrics(epochs=log_reg_model.data_for_plots['epochs'],
                                    metrics=log_reg_model.data_for_plots['accuracy_valid'],
                                    plot_title='Accuracy values on validation set',
                                    y_title='Accuracy')
    predictions_test = log_reg_model(digits_dataset(SetType.test)['inputs'])
    confusion_matrix_test = metrics.confusion_matrix(predictions_test, digits_dataset(SetType.test)['targets'])
    accuracy_test = metrics.accuracy(confusion_matrix_test)
    print('Confusion matrix on test set:')
    print(confusion_matrix_test)
    print(f'Accuracy on test set: {accuracy_test}')
    # BONUS TASK
    pickled_representation_of_log_reg_model = pickle.dumps(log_reg_model)
    return pickled_representation_of_log_reg_model


def experiment_2(digits_dataset, pickled_representation_of_log_reg_model: bytes):
    # BONUS TASK
    log_reg_model = pickle.loads(pickled_representation_of_log_reg_model)
    # BONUS TASK
    inputs = digits_dataset(SetType.valid)['inputs']
    inputs = (inputs.T * digits_dataset.std + digits_dataset.mean).T
    targets = digits_dataset(SetType.valid)['targets']
    y = log_reg_model.get_model_confidence(inputs)
    predictions = np.argmax(y, axis=0)
    maximums = np.max(y, axis=0)
    indices = maximums.argsort()[::-1]
    predictions = predictions[indices]
    inputs = inputs[indices]
    targets = targets[indices]
    right = []
    right_predictions = []
    wrong = []
    wrong_predictions = []
    i = 0
    while (len(right) < 3 or len(wrong) < 3) and i < len(predictions):
        if len(right) < 3 and predictions[i] == targets[i]:
            right.append(inputs[i])
            right_predictions.append(predictions[i])
        if len(wrong) < 3 and predictions[i] != targets[i]:
            wrong.append(inputs[i])
            wrong_predictions.append(predictions[i])
        i += 1
    Visualisation.visualise_images(images=right,
                                   predictions=right_predictions,
                                   plot_title='3 images for which classifier saved most confidence class prediction '
                                              'and was right')
    Visualisation.visualise_images(images=wrong,
                                   predictions=wrong_predictions,
                                   plot_title='3 images for which classifier saved most confidence class prediction '
                                              'and was wrong')


if __name__ == '__main__':
    digits = Digits(log_reg_cfg)
    logistic_regression_model = LogReg(log_reg_cfg, digits.k, digits.d)

    pickled = experiment(digits, logistic_regression_model)
    experiment_2(digits, pickled)

    # BONUS TASK
    logistic_regression_model = LogReg(log_reg_cfg, digits.k, digits.d)
    logistic_regression_model.batch_gradient_descent(digits(SetType.train)['inputs'], digits(SetType.train)['targets'],
                                                     digits(SetType.valid)['inputs'], digits(SetType.valid)['targets'])

    """
    # digits_dataset = Digits(log_reg_cfg)
    # print(digits_dataset.targets)
    x = np.array([[1., 3., 5.],
                  [3., 7.5, 5.],
                  [8., 12., 5.],
                  [10., 12., 5.]])
    print(10 * x)
    # targets = np.array(digits_dataset.targets).reshape(-1)
    # ohe = np.eye(10)[targets]
    # print(ohe)
    # print(ohe.shape)
    s = np.random.uniform(0, 5, (3, 7))
    print(s)
    print(np.max(x))
    print(np.amax(x))
    print(np.max(x, axis=0))
    print(np.amax(x, axis=0))
    y = np.array([[1.],
                  [3.],
                  [8.],
                  [10.]])
    x = x - np.max(x, axis=0)
    print(x)
    x = np.exp(x) / np.sum(np.exp(x), axis=0)
    print(x)
    print(x.shape[0])
    print(y[2] == 8)
    """
    """
    x = np.array([[1., 3., 5.],
                  [3., 7.5, 5.],
                  [8., 12., 5.],
                  [10., 12., 5.]])
    y = np.array([[1.],
                  [3.],
                  [8.],
                  [10.]])
    z = np.append(x, y, axis=1)
    print(np.linalg.norm(z))
    z = np.square(x)
    y = np.array([[1., 3., 5.],
                  [3., 7.5, 5.],
                  [8., 12., 5.],
                  [10., 12., 5.]])
    print(x + 100 * y)
    """
