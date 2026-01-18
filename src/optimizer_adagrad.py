import numpy as np

class Optimizer_Adagrad:

    def __init__(self, learning_rate, epsilon):
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def update_parameters(self, layer):

        if not hasattr(layer, 'cache_weights'):
            layer.cache_weights = np.zeros(layer.weights.shape)
            layer.cache_biases = np.zeros(layer.biases.shape)

        layer.cache_weights += layer.dweights ** 2
        layer.cache_biases += layer.dbiases ** 2

        layer.weights += -self.learning_rate * layer.dweights / (np.sqrt(layer.cache_weights) + self.epsilon)
        layer.biases += -self.learning_rate * layer.dbiases / (np.sqrt(layer.cache_biases) + self.epsilon)