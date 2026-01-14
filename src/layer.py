import numpy as np

class Layer:
    
    def __init__(self, inputs, n_neurons):
        self.inputs = inputs
        self.weights = self.init_weights_xavier_glorot_uniform(self.inputs.shape[1], n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self):
        self.output = np.dot(self.inputs, self.weights) + self.biases
    
    def backward(self, dvalues_es, dvalues_ao):
        self.dweights = np.dot(dvalues_es.T, dvalues_ao)

    def init_weights_xavier_glorot_uniform(self, fan_in, fan_out):
        
        # Xavier Glorot uniform initialization
        limit = np.sqrt(6 / (fan_in + fan_out))

        weights = np.random.uniform(
            low = -limit,
            high = limit,
            size = (fan_in, fan_out)
        )

        return weights





