import numpy as np

class Layer:
    
    def __init__(self, n_inputs, n_neurons):
        self.weights = self.init_weights_xavier_glorot_uniform(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def init_weights_xavier_glorot_uniform(self, fan_in, fan_out):
        
        # Xavier Glorot uniform initialization
        limit = np.sqrt(6 / (fan_in + fan_out))

        weights = np.random.uniform(
            low= -limit,
            high = limit,
            size = (fan_in, fan_out)
        )

        return weights





