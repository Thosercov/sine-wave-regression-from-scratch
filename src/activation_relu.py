import numpy as np

class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues_es, dvalues_weights):
        matrix_product = np.dot(dvalues_es, dvalues_weights.T) # sum of next layers errors signals with their respective weights
        self.error_signal = self.relu_derivative(matrix_product)

    def relu_derivative(self, dvalues):
        return dvalues * (self.output > 0).astype(int)
