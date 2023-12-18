import numpy as np

from models.decision_tree import DT
from utils.enums import TrainingAlgorithms, TaskTypes


class RandomForest:
    def __init__(self, nb_trees: int, max_depth: int, min_entropy: float, min_nb_elements: int):
        self.nb_trees = nb_trees
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_nb_elements = min_nb_elements

        self.training_algorithm = None
        self.max_nb_dim_to_check = 1
        self.max_nb_thresholds = 1
        self.subset_size = 1
        self.nb_classes = 0
        self.trees = []

    def train(self, training_algorithm, inputs: np.ndarray, targets: np.ndarray, nb_classes: int,
              max_nb_dim_to_check: int = 1, max_nb_thresholds: int = 1, subset_size: int = 1):
        if training_algorithm not in [TrainingAlgorithms.bagging, TrainingAlgorithms.random_node_optimization]:
            raise Exception('No such training algorithm! Choose bagging or random node optimization')
        self.nb_classes = nb_classes
        self.training_algorithm = training_algorithm
        self.max_nb_dim_to_check = max_nb_dim_to_check
        self.max_nb_thresholds = max_nb_thresholds
        self.subset_size = subset_size
        for i in range(self.nb_trees):
            decision_tree = DT(task_type=TaskTypes.classification,
                               max_depth=self.max_depth,
                               min_entropy=self.min_entropy,
                               min_nb_elements=self.min_nb_elements)
            if self.training_algorithm == TrainingAlgorithms.random_node_optimization:
                decision_tree.train(inputs, targets, (self.max_nb_dim_to_check, self.max_nb_thresholds))
            elif self.training_algorithm == TrainingAlgorithms.bagging:
                indices = np.random.randint(inputs.shape[0], size=self.subset_size)
                decision_tree.train(inputs[indices, :], targets[indices])
            self.trees.append(decision_tree)

    def get_predictions(self, inputs: np.ndarray) -> np.ndarray:
        if self.nb_classes == 0:
            raise Exception('The model is not trained! Before calling this method, call the train method')
        predictions = np.zeros((self.nb_trees, inputs.shape[0], self.nb_classes))
        for i in range(self.nb_trees):
            predictions[i] = self.trees[i].get_predictions(inputs, return_vector_of_confidence=True)
        return np.argmax(np.mean(predictions, axis=0), axis=1)
