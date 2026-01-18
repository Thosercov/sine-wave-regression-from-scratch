import numpy as np

class Layer:
    
    def __init__(self, n_inputs, n_neurons):
        self.weights = self.init_weights_xavier_glorot_uniform(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.dweights = np.zeros((self.weights.shape))
        self.dbiases = np.zeros((self.biases.shape))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases
    
    def backward(self, dvalues_es):
        
        #take the previous gradients and save them -> on first pass take zeros from initialization
        self.dweights_prev = self.dweights
        self.dbiases_prev = self.dbiases

        self.dweights = np.dot(dvalues_es.T, self.inputs).T #self.input is the same as Layer N-1 output -> less computation
        self.dbiases = np.sum(dvalues_es, axis=0)

    def init_weights_xavier_glorot_uniform(self, fan_in, fan_out):
        
        # Xavier Glorot uniform initialization
        limit = np.sqrt(6 / (fan_in + fan_out))

        weights = np.random.uniform(
            low = -limit,
            high = limit,
            size = (fan_in, fan_out)
        )

        return weights





