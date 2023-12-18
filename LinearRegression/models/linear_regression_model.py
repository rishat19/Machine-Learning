import numpy as np


class LinearRegression:

    def __init__(self, base_functions: list, reg_coeff: float):
        """init weights using np.random.randn (normal distribution with mean=0 and variance=1)"""
        self.weights = 0.0 + 1.0 * np.random.randn(1, len(base_functions))
        self.base_functions = base_functions
        self.reg_coeff = reg_coeff

    def __pseudoinverse_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """calculate pseudoinverse matrix with regularization using SVD"""
        u, sigma, v = np.linalg.svd(matrix)

        epsilon = np.finfo(float).eps * max(matrix.shape[0], matrix.shape[1]) * np.max(sigma)
        sigma_0 = sigma[0]
        indices = np.where(sigma <= epsilon)[0]
        sigma[indices] = 0
        indices = np.where(sigma > epsilon)[0]
        sigma[indices] = sigma[indices] / (sigma[indices] * sigma[indices] + self.reg_coeff)
        if 0 in indices:
            sigma[0] = 1 / sigma_0

        sigma_plus = np.zeros((matrix.shape[0], matrix.shape[1]))
        sigma_plus[:matrix.shape[1], :min(matrix.shape[1], matrix.shape[0])] = np.diag(sigma)
        sigma_plus = sigma_plus.T
        return np.dot(v.T, np.dot(sigma_plus, u.T))

    def __plan_matrix(self, inputs: np.ndarray) -> np.ndarray:
        """build Plan matrix using list of lambda functions defined in config. Use only one loop (for base_functions)"""
        return np.array([bf(inputs) for bf in self.base_functions]).T

    def __calculate_weights(self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray) -> None:
        """calculate weights of the model using formula from the lecture"""
        self.weights = np.dot(pseudoinverse_plan_matrix, targets.T).T

    def calculate_model_prediction(self, plan_matrix) -> np.ndarray:
        """calculate prediction of the model (y) using formula from the lecture"""
        return np.dot(self.weights, plan_matrix.T)

    def train_model(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        # prepare data
        plan_matrix = self.__plan_matrix(inputs)
        pseudoinverse_plan_matrix = self.__pseudoinverse_matrix(plan_matrix)
        # train process
        self.__calculate_weights(pseudoinverse_plan_matrix, targets)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """return prediction of the model"""
        plan_matrix = self.__plan_matrix(inputs)
        predictions = self.calculate_model_prediction(plan_matrix)
        return predictions
