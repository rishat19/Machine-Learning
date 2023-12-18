from easydict import EasyDict

from utils.enums import DataProcessTypes

cfg = EasyDict()

# data
cfg.dataframe_path = 'datasets/wine-quality-white-and-red.csv'
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.data_preprocess_type = DataProcessTypes.standardization
# cfg.data_preprocess_type = DataProcessTypes.normalization
