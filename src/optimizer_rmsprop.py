import numpy as np

class Optimizer_RMSProp:

    def __init__(self, learning_rate, rho, epsilon):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon

    def update_parameters(self, layer):

        if not hasattr(layer, 'cache_weights'):
            layer.cache_weights = np.zeros(layer.weights.shape)
            layer.cache_biases = np.zeros(layer.biases.shape)

        layer.cache_weights = self.rho * layer.cache_weights + (1 - self.rho) * layer.dweights ** 2
        layer.cache_biases = self.rho * layer.cache_biases + (1 - self.rho) * layer.dbiases ** 2

        layer.weights += -self.learning_rate * layer.dweights / (np.sqrt(layer.cache_weights) + self.epsilon)
        layer.biases += -self.learning_rate * layer.dbiases / (np.sqrt(layer.cache_biases) + self.epsilon)