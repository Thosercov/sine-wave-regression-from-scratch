import numpy as np

class Activation_Sigmoid:

    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.error_signal = dvalues * (1 - dvalues)