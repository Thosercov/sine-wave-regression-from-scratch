import numpy as np

class Activation_Sigmoid:

    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.inputs = dvalues * (1 - dvalues)