from easydict import EasyDict

from utils.common_functions import read_dataframe_file


class Titanic:
    def __init__(self, cfg: EasyDict):
        self.train_set_csv = read_dataframe_file(cfg.train_dataframe_path)
        self.test_set_csv = read_dataframe_file(cfg.test_dataframe_path)

    def __call__(self):
        return {'train_input': self.train_set_csv.values[:, 3:],
                'train_target': 2 * self.train_set_csv.values[:, 2] - 1,
                'test_input': self.test_set_csv.values[:, 3:],
                'test_target': 2 * self.test_set_csv.values[:, 2] - 1
                }
