import numpy as np

class Activation_TanH:

    def forward(self, inputs):
        self.output = (np.exp(inputs) - np.exp(-inputs))/(np.exp(inputs) + np.exp(-inputs))

    def backward(self, dvalues_es, dvalues_weights):
        matrix_product = np.dot(dvalues_es, dvalues_weights.T) # sum of next layers errors signals with their respective weights
        self.error_signal = self.tanh_derivative(matrix_product)

    def tanh_derivative(self, dvalues):
        return dvalues * (1 - self.output ** 2)