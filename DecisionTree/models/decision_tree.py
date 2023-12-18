from typing import Optional

import numpy as np

from utils.enums import TaskTypes


class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.split_index = None
        self.split_value = None
        self.terminal_node = None


class DT:
    def __init__(self, task_type, max_depth: int, min_entropy: float = 0, min_nb_elements: int = 1):
        if task_type not in [TaskTypes.classification, TaskTypes.regression]:
            raise Exception('No such task type! Choose classification or regression')
        self.task_type = task_type
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_nb_elements = min_nb_elements
        self.root = Node()

    def train(self, inputs: np.ndarray, targets: np.ndarray, random_mode: Optional[tuple] = None):
        self.__nb_dim = inputs.shape[1]
        self.__all_dim = np.arange(self.__nb_dim)
        if random_mode:
            self.__max_nb_dim_to_check, self.__max_nb_thresholds = random_mode[0], random_mode[1]  # L_1, L_2
            self.__get_axis = self.__get_random_axis
            self.__get_threshold = self.__generate_random_threshold
        else:
            self.__get_axis = self.__get_all_axis
            self.__get_threshold = self.__generate_all_threshold
        if self.task_type == TaskTypes.classification:
            self.__k = len(np.unique(targets))
            entropy_value = self.__shannon_entropy(targets, len(targets))
            self.__build_tree(inputs, targets, self.root, 1, entropy_value)
        elif self.task_type == TaskTypes.regression:
            disp_value = self.__disp(targets)
            self.__build_tree(inputs, targets, self.root, 1, disp_value)

    def __get_random_axis(self) -> np.ndarray:
        if self.__nb_dim > self.__max_nb_dim_to_check:
            return np.random.choice(self.__all_dim, size=self.__max_nb_dim_to_check, replace=False)
        else:
            return self.__all_dim

    def __get_all_axis(self) -> np.ndarray:
        return self.__all_dim

    def __create_term_array(self, targets: np.ndarray) -> np.ndarray:
        if self.task_type == TaskTypes.classification:
            term_array = np.zeros(self.__k)
            unique_classes, counts = np.unique(targets, return_counts=True)
            term_array[unique_classes] = counts / len(targets)
            return term_array
        elif self.task_type == TaskTypes.regression:
            return np.mean(targets)

    def __generate_all_threshold(self, inputs) -> np.ndarray:
        return np.unique(inputs)

    def __generate_random_threshold(self, inputs) -> np.ndarray:
        inputs = np.unique(inputs)
        if inputs.shape[0] >= self.__max_nb_thresholds:
            return np.random.choice(inputs, size=self.__max_nb_thresholds, replace=False)
        else:
            return inputs

    @staticmethod
    def __disp(targets: np.ndarray) -> float:
        return np.var(targets) if len(targets) > 0 else 0

    @staticmethod
    def __shannon_entropy(targets: np.ndarray, n: int) -> float:
        unique_classes, counts = np.unique(targets, return_counts=True)
        return -np.sum((counts / n) * np.log2(counts / n))

    def __inf_gain(self, targets_left: np.ndarray, targets_right: np.ndarray, node_entropy: float, n: int):
        left_entropy = 0
        right_entropy = 0
        if self.task_type == TaskTypes.classification:
            left_entropy = self.__shannon_entropy(targets_left, len(targets_left))
            right_entropy = self.__shannon_entropy(targets_right, len(targets_right))
        elif self.task_type == TaskTypes.regression:
            left_entropy = self.__disp(targets_left)
            right_entropy = self.__disp(targets_right)
        information_gain = node_entropy - len(targets_left) * left_entropy / n - len(targets_right) * right_entropy / n
        return information_gain, left_entropy, right_entropy

    def __build_splitting_node(self, inputs: np.ndarray, targets: np.ndarray, entropy: float, n: int):
        split_index = 0
        split_value = 0
        entropy_left_max = 0
        entropy_right_max = 0
        indices_left_max = None
        indices_right_max = None
        information_gain_max = -1
        for axis in self.__get_axis():
            for threshold in self.__get_threshold(inputs[:, axis]):
                indices_left = inputs[:, axis] <= threshold
                indices_right = ~indices_left
                information_gain, entropy_left, entropy_right = \
                    self.__inf_gain(targets[indices_left], targets[indices_right], entropy, n)
                if information_gain > information_gain_max:
                    split_index = axis
                    split_value = threshold
                    entropy_left_max = entropy_left
                    entropy_right_max = entropy_right
                    indices_left_max = indices_left
                    indices_right_max = indices_right
                    information_gain_max = information_gain
        return split_index, split_value, indices_left_max, indices_right_max, entropy_left_max, entropy_right_max

    def __build_tree(self, inputs: np.ndarray, targets: np.ndarray, node: Node, depth: int, entropy: float):
        n = len(targets)
        if depth >= self.max_depth or entropy <= self.min_entropy or n <= self.min_nb_elements:
            node.terminal_node = self.__create_term_array(targets)
        else:
            split_index, split_value, indices_left, indices_right, entropy_left, entropy_right = \
                self.__build_splitting_node(inputs, targets, entropy, n)
            node.split_index = split_index
            node.split_value = split_value
            node.left_child = Node()
            node.right_child = Node()
            self.__build_tree(inputs[indices_left], targets[indices_left], node.left_child, depth + 1, entropy_left)
            self.__build_tree(inputs[indices_right], targets[indices_right], node.right_child, depth + 1, entropy_right)

    def get_predictions(self, inputs: np.ndarray, return_vector_of_confidence: bool = False) -> np.ndarray:
        node = self.root
        results = np.zeros(inputs.shape[0])
        if return_vector_of_confidence and self.task_type == TaskTypes.classification:
            results = np.zeros((inputs.shape[0], self.__k))
        for i in range(inputs.shape[0]):
            while node.terminal_node is None:
                if inputs[i][node.split_index] > node.split_value:
                    node = node.right_child
                else:
                    node = node.left_child
            if self.task_type == TaskTypes.regression or return_vector_of_confidence:
                results[i] = node.terminal_node
            elif self.task_type == TaskTypes.classification:
                results[i] = np.argmax(node.terminal_node)
            node = self.root
        return results
