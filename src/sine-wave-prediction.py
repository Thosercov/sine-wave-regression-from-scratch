import numpy as np
import matplotlib.pyplot as plt
import constants as c
from layer import Layer
from activation_sigmoid import Activation_Sigmoid
from activation_linear import Activation_linear
from loss_mse import Loss_MSE
from optimizer_sgd import Optimizer_SGD
from optimizer_sgd_decay import Optimizer_SGD_Decay
from optimizer_sgd_momentum import Optimizer_SGD_Momentum
from optimizer_adagrad import Optimizer_Adagrad
from optimizer_rmsprop import Optimizer_RMSProp
from optimizer_adam import Optimizer_Adam

np.random.seed(0)


x_samples = np.random.uniform(low = 0.0, high = 2 * np.pi, size = (c.N_SAMPLES, 1))
y_samples = np.sin(x_samples) + np.random.normal(loc = 0.0, scale = 0.3, size = (c.N_SAMPLES, 1))

optimizer_sgd = Optimizer_SGD(c.LEARNING_RATE)
optimizer_sgd_decay = Optimizer_SGD_Decay(c.LEARNING_RATE, c.STEP, c.LEARNING_RATE_DECAY)
optimizer_sgd_momentum = Optimizer_SGD_Momentum(c.LEARNING_RATE, c.MOMENTUM_BETA)
optimizer_adagrad = Optimizer_Adagrad(c.LEARNING_RATE, c.EPSILON)
optimizer_rmsprop = Optimizer_RMSProp(c.LEARNING_RATE, c.RHO, c.EPSILON)
optimizer_adam = Optimizer_Adam(c.LEARNING_RATE, c.ADAM_BETA_1, c.ADAM_BETA_2, c.EPSILON)

layer1 = Layer(c.N_FEATURES, c.N_NEURONS_L1)
layer2 = Layer(c.N_NEURONS_L1, c.N_NEURONS_L2)
layer3 = Layer(c.N_NEURONS_L2, c.N_NEURONS_L3)
layer_output = Layer(c.N_NEURONS_L3, c.N_NEURONS_OUTPUT)

activation1 = Activation_Sigmoid()
activation2 = Activation_Sigmoid()
activation3 = Activation_Sigmoid()
activation_output = Activation_linear()

for i in range(c.N_EPOCHS):

    layer1.forward(x_samples)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    layer3.forward(activation2.output)
    activation3.forward(layer3.output)

    layer_output.forward(activation3.output)
    activation_output.forward(layer_output.output)

    loss = Loss_MSE(activation_output.output, y_samples)
    loss.forward()

    print("Pass: ", i, " Loss: ", loss.output)

    # backward pass of the data

    loss.backward()
    activation_output.backward(loss.d_loss)
    layer_output.backward(activation_output.error_signal)

    activation3.backward(activation_output.error_signal, layer_output.weights)
    layer3.backward(activation3.error_signal)

    activation2.backward(activation3.error_signal, layer3.weights)
    layer2.backward(activation2.error_signal)

    activation1.backward(activation2.error_signal, layer2.weights)
    layer1.backward(activation1.error_signal)

    optimizer_sgd_momentum.update_parameters(layer1)
    optimizer_sgd_momentum.update_parameters(layer2)
    optimizer_sgd_momentum.update_parameters(layer3)
    optimizer_sgd_momentum.update_parameters(layer_output)


plt.scatter(x_samples, y_samples)
plt.scatter(x_samples, activation_output.output)
plt.show()