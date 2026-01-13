import numpy as np
import matplotlib.pyplot as plt
import constants as c
from layer import Layer
from activation_sigmoid import Activation_Sigmoid
from activation_linear import Activation_linear
from loss_mse import Loss_MSE


np.random.seed(0)


x_samples = np.random.uniform(low = 0.0, high = 2 * np.pi, size = (c.N_SAMPLES, 1))
y_samples = np.sin(x_samples) + np.random.normal(loc = 0.0, scale = 0.3, size = (c.N_SAMPLES, 1))

# forward pass of the data

layer1 = Layer(x_samples, c.N_NEURONS_L1)
layer1.forward()
activation1 = Activation_Sigmoid()
activation1.forward(layer1.output)


layer2 = Layer(activation1.output, c.N_NEURONS_L2)
layer2.forward()
activation2 = Activation_Sigmoid()
activation2.forward(layer2.output)

layer3 = Layer(activation2.output, c.N_NEURONS_L3)
layer3.forward()
activation3 = Activation_Sigmoid()
activation3.forward(layer3.output)

layer_output = Layer(activation3.output, c.N_NEURONS_OUTPUT)
layer_output.forward()
activation_output = Activation_linear()
activation_output.forward(layer_output.output)

loss = Loss_MSE(activation_output.output, y_samples)
loss.forward()

# backward pass of the data

loss.backward()
activation_output.backward(loss.d_loss)
#plt.scatter(x_samples, y_samples)
#plt.scatter(x_samples, activation_output.output)
#plt.show()

print(layer3.output.shape)