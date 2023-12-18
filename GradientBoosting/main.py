import numpy as np

import utils.metrics as metrics
from config.gradient_boosting_config import cfg as gradient_boosting_config
from datasets.wine_quality_dataset import WineQuality
from models.gradient_boosting import GradientBoosting
from utils.enums import SetType
from utils.visualisation import Visualisation


def experiment_main_task(wine_quality: WineQuality):
    gradient_boosting = GradientBoosting(number_of_weak_learners=100,
                                         weight_of_weak_learners=0.05)
    gradient_boosting.train(inputs=wine_quality(SetType.train)['inputs'],
                            targets=wine_quality(SetType.train)['targets'])
    predictions_test = gradient_boosting.get_predictions(wine_quality(SetType.test)['inputs'])
    mean_squared_error_test = metrics.MSE(predictions_test, wine_quality(SetType.test)['targets'])
    print(f'Error value on test set: {mean_squared_error_test}')


def experiment_additional_task(wine_quality: WineQuality):
    # validation
    number_of_models = 30
    models = []
    for i in range(number_of_models):
        m = np.random.randint(low=10, high=51)
        alpha = np.random.uniform(low=0.01, high=0.1)
        gradient_boosting = GradientBoosting(number_of_weak_learners=m,
                                             weight_of_weak_learners=alpha)
        gradient_boosting.train(inputs=wine_quality(SetType.train)['inputs'],
                                targets=wine_quality(SetType.train)['targets'])
        predictions_valid = gradient_boosting.get_predictions(wine_quality(SetType.valid)['inputs'])
        mean_squared_error_valid = metrics.MSE(predictions_valid, wine_quality(SetType.test)['targets'])
        print(f'Validation: model #{i + 1}, M = {m}, alpha = {alpha}, MSE = {mean_squared_error_valid}')
        models.append([gradient_boosting, mean_squared_error_valid, 0])

    # testing best models
    models.sort(key=lambda x: x[1])
    best_models = models[:10]
    for i in range(len(best_models)):
        predictions_test = best_models[i][0].get_predictions(wine_quality(SetType.test)['inputs'])
        mean_squared_error_test = metrics.MSE(predictions_test, wine_quality(SetType.test)['targets'])
        best_models[i][2] = mean_squared_error_test

    # plot for best models
    Visualisation.visualise_best_models(best_models, plot_title='10 best models')


if __name__ == '__main__':
    wq = WineQuality(gradient_boosting_config)
    experiment_main_task(wq)
    experiment_additional_task(wq)
