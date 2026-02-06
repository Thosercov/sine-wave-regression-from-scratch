import numpy as np

class Loss_MSE:

    def calculate(self, y_prediction, y_true):
        delta = y_prediction - y_true
        return 0.5 * np.mean(delta ** 2)
    
    def backward(self, y_prediction, y_true):
        delta = y_prediction- y_true
        n = delta.size
        self.d_loss= delta / n

    def regularization_loss(self, layer):

        regularization_loss = 0

        # L1 regularization
        if layer.weight_lambda_l1 > 0:
            regularization_loss += layer.weight_lambda_l1 * np.sum(np.abs(layer.weights))

        if layer.bias_lambda_l1 > 0:
            regularization_loss += layer.bias_lambda_l1 * np.sum(np.abs(layer.weights))

        # L1 regularization
        if layer.weight_lambda_l2 > 0:
            regularization_loss += layer.weight_lambda_l2 * np.sum(layer.weights ** 2)

        if layer.bias_lambda_l2 > 0:
            regularization_loss += layer.bias_lambda_l2 * np.sum(layer.weights ** 2)
        

        return regularization_loss