import numpy as np
class Optimizer_SGD_Momentum:

    def __init__(self, learning_rate, beta):
        self.learning_rate = learning_rate
        self.beta = beta

    def update_parameters(self, layer):

        #1st pass take zeroes
        if not hasattr(layer, 'dweights_prev'):
            layer.dweights_prev = np.zeros(layer.weights.shape)
            layer.dbiases_prev = np.zeros(layer.biases.shape)

        velocity_weights = self.beta * layer.dweights_prev +  self.learning_rate * layer.dweights
        velocity_biases = self.beta * layer.dbiases_prev + self.learning_rate * layer.dbiases

        # update weights by subtracting the negative gradients multiplied with learning rate
        layer.weights += -velocity_weights
        layer.biases += -velocity_biases

        #set current biases as previous for the next pass
        layer.dweights_prev = layer.dweights
        layer.dbiases_prev = layer.dbiases