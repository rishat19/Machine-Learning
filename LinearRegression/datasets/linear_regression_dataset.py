import numpy as np
from utils.common_functions import read_dataframe_file
from easydict import EasyDict


class LinRegDataset:

    def __init__(self, cfg: EasyDict):
        advertising_dataframe = read_dataframe_file(cfg.dataframe_path)
        advertising_dataframe = advertising_dataframe.sample(frac=1).reset_index(drop=True)  # shuffle the dataframe
        inputs, targets = np.asarray(advertising_dataframe['inputs']), np.asarray(advertising_dataframe['targets'])
        self.__divide_into_sets(inputs, targets, cfg.train_set_percent, cfg.valid_set_percent)

    def __divide_into_sets(self, inputs: np.ndarray, targets: np.ndarray, train_set_percent: float = 0.8,
                           valid_set_percent: float = 0.1) -> None:
        index_1 = int(inputs.size * train_set_percent)
        index_2 = int(inputs.size * (train_set_percent + valid_set_percent))
        self.inputs_train, self.inputs_valid, self.inputs_test = np.split(inputs, [index_1, index_2])
        self.targets_train, self.targets_valid, self.targets_test = np.split(targets, [index_1, index_2])

    def __call__(self) -> dict:
        return {'inputs': {'train': self.inputs_train,
                           'valid': self.inputs_valid,
                           'test': self.inputs_test},
                'targets': {'train': self.targets_train,
                            'valid': self.targets_valid,
                            'test': self.targets_test}
                }
