class Optimizer_BGD_Decay:

    def __init__(self, learning_rate, step, learning_rate_decay):
        self.step = step
        self.learning_rate_decay = learning_rate_decay
        self.start_learning_rate = learning_rate

    def update_parameters(self, layer):
        # multiply the start learning rate  with the fraction of learning rate decay which keeps decreasing with steps
        self.learning_rate = self.start_learning_rate * (1 / (1 + self.learning_rate_decay * self.step)) 

        # update weights by subtracting the negative gradients multiplied with learning rate
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

        self.step += 1

