import numpy as np

class Layer:
    
    def __init__(self, n_inputs, n_neurons,
                 weight_lambda_l1=0, weight_lambda_l2=0,
                 bias_lambda_l1=0, bias_lambda_l2=0):
        
        self.weights = self.init_weights_xavier_glorot_uniform(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.weight_lambda_l1 = weight_lambda_l1
        self.weight_lambda_l2 = weight_lambda_l2
        self.bias_lambda_l1 = bias_lambda_l1
        self.bias_lambda_l2 = bias_lambda_l2


    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases
    
    def backward(self, dvalues_es):
        self.dweights = np.dot(self.inputs.T, dvalues_es) #self.input is the same as Layer N-1 output -> less computation
        self.dbiases = np.sum(dvalues_es, axis=0, keepdims=1)

        if self.weight_lambda_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_lambda_l1 * dL1

        if self.bias_lambda_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1 
            self.dbiases += self.bias_lambda_l1 * dL1

        if self.weight_lambda_l2 > 0:
            self.dweights += 2 * self.weight_lambda_l2 * self.weights

        if self.bias_lambda_l2 > 0:
            self.dweights += 2 * self.bias_lambda_l2 * self.weights

    def init_weights_xavier_glorot_uniform(self, fan_in, fan_out):
        # Xavier Glorot uniform initialization
        limit = np.sqrt(6 / (fan_in + fan_out))

        weights = np.random.uniform(
            low = -limit,
            high = limit,
            size = (fan_in, fan_out)
        )

        return weights
    
    def init_weights_kaiming_he_uniform(self, fan_in, fan_out):
        # Kaiming He uniform initialization
        limit = np.sqrt(6 / fan_in)

        weights = np.random.uniform(
            low = -limit,
            high = limit,
            size = (fan_in, fan_out)
        )

        return weights





