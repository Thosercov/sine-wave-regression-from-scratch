class Activation_linear:

    def forward(self, inputs):
        self.output = inputs

    def backward(self, dvalues):
        self.error_signal = dvalues.copy()