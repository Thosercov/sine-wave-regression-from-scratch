class Optimizer_SGD:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_parameters(self, layer):

        # update weights by subtracting the negative gradients multiplied with learning rate
        layer.weights += -self.learning_rate * layer.dweights.T 
        layer.biases += -self.learning_rate * layer.dbiases