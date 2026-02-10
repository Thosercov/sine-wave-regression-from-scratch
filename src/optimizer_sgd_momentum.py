import numpy as np
class Optimizer_SGD_Momentum:

    def __init__(self, learning_rate, beta):
        self.learning_rate = learning_rate
        self.beta = beta

    def update_parameters(self, layer):

        # 1st pass take zeroes
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros(layer.weights.shape)
            layer.bias_momentums = np.zeros(layer.biases.shape)

        velocity_weights = self.beta * layer.weight_momentums - self.learning_rate * layer.dweights
        velocity_biases = self.beta * layer.bias_momentums - self.learning_rate * layer.dbiases

        layer.weights += velocity_weights
        layer.biases += velocity_biases

        layer.weight_momentums = velocity_weights
        layer.bias_momentums = velocity_biases