from models.linear_regression_model import LinearRegression
from datasets.linear_regression_dataset import LinRegDataset
from utils.metrics import MSE
from utils.visualisation import Visualisation
import numpy as np


def search_for_random_hyperparameters(smallest_max_degree: int = 5,
                                      biggest_max_degree: int = 200,
                                      smallest_reg_coeff: float = 0.0,
                                      biggest_reg_coeff: float = 5.0,
                                      number_of_hyperparameter_sets: int = 100):
    return np.random.randint(low=smallest_max_degree, high=biggest_max_degree + 1, size=number_of_hyperparameter_sets), \
        np.random.uniform(low=smallest_reg_coeff, high=biggest_reg_coeff, size=number_of_hyperparameter_sets)


def experiment(lin_reg_cfg, lin_reg_dataset, visualise_prediction=True):
    """find 10 best model according to error on validation set; make plot with 10 points, where x-axis name of the
    model (max degree + regularisation coefficient), y-axis error on valid set; add to hover_data error on test set"""

    # training
    lin_reg_model = LinearRegression(lin_reg_cfg.base_functions, 2)
    lin_reg_model.train_model(lin_reg_dataset()['inputs']['train'], lin_reg_dataset()['targets']['train'])

    # validation
    max_degrees, reg_coefficients = search_for_random_hyperparameters()
    results = []
    for i in range(max_degrees.size):
        lin_reg_cfg.update(base_functions=[lambda x, arg=i: pow(x, arg) for i in range(max_degrees[i] + 1)])
        lin_reg_model = LinearRegression(lin_reg_cfg.base_functions, reg_coefficients[i])
        lin_reg_model.train_model(lin_reg_dataset()['inputs']['valid'], lin_reg_dataset()['targets']['valid'])
        predictions = lin_reg_model(lin_reg_dataset()['inputs']['valid'])
        error = MSE(predictions, lin_reg_dataset()['targets']['valid'])
        results.append([max_degrees[i], reg_coefficients[i], error, lin_reg_model])
    results.sort(key=lambda x: x[2])
    results = results[:10]

    # testing
    models = []
    valid_errors = []
    test_errors = []
    for i in range(len(results)):
        lin_reg_model = results[i][3]
        predictions = lin_reg_model(lin_reg_dataset()['inputs']['test'])
        error = MSE(predictions, lin_reg_dataset()['targets']['test'])
        models.append(f'max_degree={results[i][0]} + reg_coeff={round(results[i][1], 3)}')
        valid_errors.append(str(results[i][2]))
        test_errors.append(str(error))
    if visualise_prediction:
        Visualisation.visualise_best_models(models, valid_errors, test_errors, '10 best models')


def experiment_2(lin_reg_cfg, lin_reg_dataset, max_degree: int, reg_coeff: float, visualise_prediction=True):
    """make plots with two traces - model prediction and target values; target trace need to be in markers mode,
    model prediction - line; make plot for model with or without regularisation max degree of polynomials 100"""
    lin_reg_cfg.update(base_functions=[lambda x, arg=i: pow(x, arg) for i in range(max_degree + 1)])
    lin_reg_model = LinearRegression(lin_reg_cfg.base_functions, reg_coeff)
    lin_reg_model.train_model(lin_reg_dataset()['inputs']['train'], lin_reg_dataset()['targets']['train'])
    predictions = lin_reg_model(lin_reg_dataset()['inputs']['test'])
    error = MSE(predictions, lin_reg_dataset()['targets']['test'])
    if visualise_prediction:
        Visualisation.visualise_predicted_trace(predictions,
                                                lin_reg_dataset()['inputs']['test'],
                                                lin_reg_dataset()['targets']['test'],
                                                plot_title=f'Полином степени {max_degree}; '
                                                           f'коэффициент регуляризации = {reg_coeff}')


if __name__ == '__main__':
    from configs.linear_regression_cfg import cfg as lin_reg_cfg

    lin_reg_dataset = LinRegDataset(lin_reg_cfg)
    experiment(lin_reg_cfg, lin_reg_dataset, visualise_prediction=True)
    experiment_2(lin_reg_cfg, lin_reg_dataset, 100, 1e-5, visualise_prediction=True)
    experiment_2(lin_reg_cfg, lin_reg_dataset, 100, 0, visualise_prediction=True)
