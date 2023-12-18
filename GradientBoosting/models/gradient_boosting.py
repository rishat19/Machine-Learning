import numpy as np

from models.decision_tree import DT
from utils.enums import TaskTypes


class GradientBoosting:
    def __init__(self, number_of_weak_learners: int, weight_of_weak_learners: float):
        self.number_of_weak_learners = number_of_weak_learners
        self.weight_of_weak_learners = weight_of_weak_learners
        self.__zero_learner = None
        self.__weak_learners = []

    def __init_zero_learner(self, targets: np.ndarray) -> np.ndarray:
        self.__zero_learner = np.mean(targets)
        return np.full(shape=len(targets), fill_value=self.__zero_learner)

    def train(self, inputs: np.ndarray, targets: np.ndarray):
        y = self.__init_zero_learner(targets)
        for i in range(self.number_of_weak_learners):
            residuals = targets - y
            decision_stump = DT(task_type=TaskTypes.regression,
                                max_depth=1,
                                min_entropy=0.01,
                                min_nb_elements=1)
            decision_stump.train(inputs=inputs,
                                 targets=residuals)
            predictions = decision_stump.get_predictions(inputs)
            self.__weak_learners.append(decision_stump)
            y = y + self.weight_of_weak_learners * predictions

    def get_predictions(self, inputs: np.ndarray) -> np.ndarray:
        if len(self.__weak_learners) == 0:
            raise Exception('The model is not trained! Before calling this method, call the train method')
        predictions = np.full(shape=inputs.shape[0], fill_value=self.__zero_learner)
        for i in range(self.number_of_weak_learners):
            predictions += (self.weight_of_weak_learners * self.__weak_learners[i].get_predictions(inputs))
        return predictions
