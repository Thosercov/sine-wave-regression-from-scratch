class Optimizer_SGD_Momentum:

    def __init__(self, learning_rate, beta):
        self.learning_rate = learning_rate
        self.beta = beta

    def update_parameters(self, layer):

        velocity_weights = self.beta * layer.dweights_prev +  self.learning_rate * layer.dweights
        velocity_biases = self.beta * layer.dbiases_prev + self.learning_rate * layer.dbiases

        # update weights by subtracting the negative gradients multiplied with learning rate
        layer.weights += -velocity_weights
        layer.biases += -velocity_biases