import numpy as np

class Loss_MSE:
    def calculate(self, y_prediction, y_true):
        delta = y_prediction - y_true
        return 0.5 * np.mean(delta ** 2)