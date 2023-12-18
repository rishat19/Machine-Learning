import numpy as np

import utils.metrics as metrics
from config.decision_tree_config import cfg as des_tree_cfg
from datasets.digits_dataset import Digits
from datasets.wine_quality_dataset import WineQuality
from models.decision_tree import DT
from models.random_forest import RandomForest
from utils.enums import SetType, TaskTypes, TrainingAlgorithms
from utils.visualisation import Visualisation


def experiment_classification(digits: Digits):
    decision_tree = DT(task_type=TaskTypes.classification,
                       max_depth=7,
                       min_entropy=0.01,
                       min_nb_elements=1)
    decision_tree.train(digits(SetType.train)['inputs'], digits(SetType.train)['targets'])

    predictions_valid = decision_tree.get_predictions(digits(SetType.valid)['inputs'])
    confusion_matrix_valid = metrics.confusion_matrix(predictions_valid, digits(SetType.valid)['targets'])
    accuracy_valid = metrics.accuracy(confusion_matrix_valid)
    print('Confusion matrix on valid set:')
    print(confusion_matrix_valid)
    print(f'Accuracy on valid set: {accuracy_valid}')

    print()

    predictions_test = decision_tree.get_predictions(digits(SetType.test)['inputs'])
    confusion_matrix_test = metrics.confusion_matrix(predictions_test, digits(SetType.test)['targets'])
    accuracy_test = metrics.accuracy(confusion_matrix_test)
    print('Confusion matrix on test set:')
    print(confusion_matrix_test)
    print(f'Accuracy on test set: {accuracy_test}')


def experiment_regression():
    wine_quality = WineQuality(des_tree_cfg)
    decision_tree = DT(task_type=TaskTypes.regression,
                       max_depth=10,
                       min_entropy=0.01,
                       min_nb_elements=1)
    decision_tree.train(wine_quality(SetType.train)['inputs'], wine_quality(SetType.train)['targets'])

    predictions_valid = decision_tree.get_predictions(wine_quality(SetType.valid)['inputs'])
    mean_squared_error_valid = metrics.MSE(predictions_valid, wine_quality(SetType.valid)['targets'])
    print(f'Error value on valid set: {mean_squared_error_valid}')

    predictions_test = decision_tree.get_predictions(wine_quality(SetType.test)['inputs'])
    # targets_test = wine_quality(SetType.test)['targets']
    # for i in range(len(targets_test)):
    #     print(f'{targets_test[i]} {predictions_test[i]}')
    mean_squared_error_test = metrics.MSE(predictions_test, wine_quality(SetType.test)['targets'])
    print(f'Error value on test set: {mean_squared_error_test}')


def experiment_random_forest(digits: Digits):
    # validation
    nb_models = 30
    models = []
    for i in range(nb_models):
        l_1 = np.random.randint(2, 3)
        l_2 = np.random.randint(5, 35)
        m = np.random.randint(5, 20)
        random_forest = RandomForest(nb_trees=m,
                                     max_depth=7,
                                     min_entropy=0.01,
                                     min_nb_elements=1)
        random_forest.train(training_algorithm=TrainingAlgorithms.random_node_optimization,
                            inputs=digits(SetType.train)['inputs'],
                            targets=digits(SetType.train)['targets'],
                            nb_classes=digits.k,
                            max_nb_dim_to_check=l_1,
                            max_nb_thresholds=l_2)
        predictions_valid = random_forest.get_predictions(digits(SetType.valid)['inputs'])
        confusion_matrix_valid = metrics.confusion_matrix(predictions_valid, digits(SetType.valid)['targets'])
        accuracy_valid = metrics.accuracy(confusion_matrix_valid)
        print(f'Validation: model #{i + 1}, M = {m}, L1 = {l_1}, L2 = {l_2}, accuracy_valid = {accuracy_valid}')
        models.append([random_forest, accuracy_valid, 0])

    # testing best models
    models.sort(key=lambda x: x[1], reverse=True)
    best_models = models[:10]
    for i in range(len(best_models)):
        predictions_test = best_models[i][0].get_predictions(digits(SetType.test)['inputs'])
        confusion_matrix_test = metrics.confusion_matrix(predictions_test, digits(SetType.test)['targets'])
        accuracy_test = metrics.accuracy(confusion_matrix_test)
        best_models[i][2] = accuracy_test

    # plot for best models
    Visualisation.visualise_best_models(best_models, plot_title='10 best models')

    # confusion matrix for best model
    best_model = best_models[0][0]
    predictions_test = best_model.get_predictions(digits(SetType.test)['inputs'])
    confusion_matrix_test = metrics.confusion_matrix(predictions_test, digits(SetType.test)['targets'])
    print()
    print('Confusion matrix for best model on test set:')
    print(confusion_matrix_test)


# BONUS TASK
def experiment_random_forest_bagging(digits: Digits):
    random_forest = RandomForest(nb_trees=10,
                                 max_depth=7,
                                 min_entropy=0.01,
                                 min_nb_elements=1)
    random_forest.train(training_algorithm=TrainingAlgorithms.bagging,
                        inputs=digits(SetType.train)['inputs'],
                        targets=digits(SetType.train)['targets'],
                        nb_classes=digits.k,
                        subset_size=int(digits(SetType.train)['inputs'].shape[0] * 0.8))
    predictions_test = random_forest.get_predictions(digits(SetType.test)['inputs'])
    confusion_matrix_test = metrics.confusion_matrix(predictions_test, digits(SetType.test)['targets'])
    accuracy_test = metrics.accuracy(confusion_matrix_test)
    print('BAGGING. Confusion matrix on test set:')
    print(confusion_matrix_test)
    print(f'BAGGING. Accuracy on test set: {accuracy_test}')


if __name__ == '__main__':
    digits_dataset = Digits(des_tree_cfg)
    experiment_classification(digits_dataset)
    print()
    experiment_regression()
    print()
    experiment_random_forest(digits_dataset)
    print()
    experiment_random_forest_bagging(digits_dataset)

    """
    A = np.random.randint(5, size=(7, 5, 3))
    print(A)
    B = np.mean(A, axis=0)
    print(B)
    print(np.argmax(B, axis=1))
    """

    """
    digits = Digits(log_reg_cfg)
    logistic_regression_model = LogReg(log_reg_cfg, digits.k, digits.d)

    pickled = experiment(digits, logistic_regression_model)
    experiment_2(digits, pickled)

    # BONUS TASK
    logistic_regression_model = LogReg(log_reg_cfg, digits.k, digits.d)
    logistic_regression_model.batch_gradient_descent(digits(SetType.train)['inputs'], digits(SetType.train)['targets'],
                                                     digits(SetType.valid)['inputs'], digits(SetType.valid)['targets'])

    """

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
