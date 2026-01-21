import numpy as np

class Activation_Leaky_ReLU:

    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, inputs):
        self.output = np.where(inputs > 0, inputs, inputs * 0.01)

    def backward(self, dvalues_es, dvalues_weights):
        matrix_product = np.dot(dvalues_es, dvalues_weights.T) # sum of next layers errors signals with their respective weights
        self.error_signal = self.relu_derivative(matrix_product)

    def relu_derivative(self, dvalues):
        return dvalues * np.where(self.output > 0, self.output , self.alpha)
