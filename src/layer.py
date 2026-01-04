import numpy as np

class Layer:
    
    def __init__(self, n_inputs, n_neurons):
        self.initialize_weights(n_inputs, n_neurons)

    def initialize_weights(self, fan_in, fan_out):
        
        # Xavier Glorot uniform initialization
        limit = np.sqrt(6 / (fan_in + fan_out))

        weights = np.random.uniform(
            low= -limit,
            high = limit,
            size = (fan_in, fan_out)
        )

        self.weights = weights





