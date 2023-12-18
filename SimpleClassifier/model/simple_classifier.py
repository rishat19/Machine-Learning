from config.cfg import cfg
import numpy as np

class Classifier:
    def __call__(self, height):
        """returns confidence of belonging to the class of basketball players"""
        return np.round(height/cfg.max_height, 5)
