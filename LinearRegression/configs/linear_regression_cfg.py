from easydict import EasyDict
import numpy as np

cfg = EasyDict()
cfg.dataframe_path = 'linear_regression_dataset.csv'
cfg.base_functions = [lambda x, arg=i: pow(x, arg) for i in range(8 + 1)]
cfg.regularization_coeff = 0
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1

# [lambda x: np.array([pow(x, i) for i in range(2)]),
#  lambda x: np.array([pow(x, i) for i in range(9)]),
#  lambda x: np.array([pow(x, i) for i in range(101)])]
