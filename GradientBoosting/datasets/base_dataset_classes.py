from abc import ABC, abstractmethod

import numpy as np


class BaseDataset(ABC):
    def __init__(self, train_set_percent, valid_set_percent):
        self.targets_test = None
        self.targets_valid = None
        self.targets_train = None
        self.inputs_test = None
        self.inputs_valid = None
        self.inputs_train = None
        self.train_set_percent = train_set_percent
        self.valid_set_percent = valid_set_percent

    @property
    @abstractmethod
    def targets(self):
        # targets variables
        pass

    @property
    @abstractmethod
    def inputs(self):
        # inputs variables
        pass

    @property
    @abstractmethod
    def d(self):
        # inputs variables
        pass

    def divide_into_sets(self):
        randomize = np.arange(self.inputs.shape[0])
        np.random.shuffle(randomize)
        inputs = self.inputs[randomize]
        targets = self.targets[randomize]
        index_1 = int(inputs.shape[0] * self.train_set_percent)
        index_2 = int(inputs.shape[0] * (self.train_set_percent + self.valid_set_percent))
        self.inputs_train, self.inputs_valid, self.inputs_test = np.split(inputs, [index_1, index_2])
        self.targets_train, self.targets_valid, self.targets_test = np.split(targets, [index_1, index_2])

    def normalization(self):
        # BONUS TASK
        inputs_min = np.min(self.inputs, axis=0).reshape(self.inputs.shape[1], 1)
        inputs_max = np.max(self.inputs, axis=0).reshape(self.inputs.shape[1], 1)
        difference = inputs_max - inputs_min
        difference[difference == 0] = 1
        self.inputs_train = ((self.inputs_train.T - inputs_min) / difference).T
        self.inputs_valid = ((self.inputs_valid.T - inputs_min) / difference).T
        self.inputs_test = ((self.inputs_test.T - inputs_min) / difference).T

    def __get_data_stats(self):
        mean = np.mean(self.inputs_train, axis=0).reshape(self.inputs_train.shape[1], 1)
        std = np.std(self.inputs_train, axis=0).reshape(self.inputs_train.shape[1], 1)
        std[std == 0] = 1
        return mean, std

    def standardization(self):
        mean, std = self.__get_data_stats()
        self.inputs_train = ((self.inputs_train.T - mean) / std).T
        self.inputs_valid = ((self.inputs_valid.T - mean) / std).T
        self.inputs_test = ((self.inputs_test.T - mean) / std).T
        return mean, std


class BaseClassificationDataset(BaseDataset):

    @property
    @abstractmethod
    def k(self):
        # number of classes
        pass

    @staticmethod
    def onehotencoding(targets: np.ndarray, number_classes: int) -> np.ndarray:
        targets = np.array(targets).reshape(-1)
        return np.eye(number_classes)[targets]
