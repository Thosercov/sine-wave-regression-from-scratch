import numpy as np
class Optimizer_Adam:

    def __init__(self, learning_rate, beta_1, beta_2, epsilon):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0

    def update_parameters(self, layer):

        #1st pass take zeroes
        if not hasattr(layer, 'cache_weights'):
            layer.weight_momentums = np.zeros(layer.weights.shape)
            layer.bias_momentums = np.zeros(layer.biases.shape)
            layer.cache_weights = np.zeros(layer.weights.shape)
            layer.cache_biases = np.zeros(layer.biases.shape)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.cache_weights = self.beta_2 * layer.cache_weights + (1 - self.beta_2) * layer.dweights**2
        layer.cache_biases = self.beta_2 * layer.cache_biases + (1 - self.beta_2) * layer.dbiases**2

        weight_cache_corrected = layer.cache_weights / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.cache_biases / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) +self.epsilon)
        layer.biases += -self.learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)