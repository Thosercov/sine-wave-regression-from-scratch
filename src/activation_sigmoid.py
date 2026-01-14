import numpy as np

class Activation_Sigmoid:

    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues_es, dvalues_weights):
        matrix_product = np.dot(dvalues_es, dvalues_weights.T) # sum of next layers errors signals with their respective weights
        self.error_signal = self.sigmoid_derivative(matrix_product)

    def sigmoid_derivative(self, dvalues):
        return dvalues * self.output * (1 - self.output)