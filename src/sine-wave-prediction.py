import numpy as np
import matplotlib.pyplot as plt
import constants as c
from layer import Layer
from activation_sigmoid import Activation_Sigmoid
from activation_linear import Activation_linear
from activation_tanh import Activation_TanH
from activation_relu import Activation_ReLU
from loss_mse import Loss_MSE
from optimizer_sgd import Optimizer_SGD
from optimizer_sgd_decay import Optimizer_SGD_Decay
from optimizer_sgd_momentum import Optimizer_SGD_Momentum
from optimizer_adagrad import Optimizer_Adagrad
from optimizer_rmsprop import Optimizer_RMSProp
from optimizer_adam import Optimizer_Adam
from activation_leaky_relu import Activation_Leaky_ReLU

np.random.seed(0)


x_samples = np.random.uniform(low = 0.0, high = 2 * np.pi, size = (c.N_SAMPLES, 1))
y_samples = np.sin(x_samples) + np.random.normal(loc = 0.0, scale = 0.3, size = (c.N_SAMPLES, 1))

idx = np.random.permutation(c.N_SAMPLES)
split = int(c.TRAINING_DATA_RATIO * c.N_SAMPLES)

train_idx = idx[:split]
val_idx   = idx[split:]

x_train, y_train = x_samples[train_idx], y_samples[train_idx]
x_val,   y_val   = x_samples[val_idx],   y_samples[val_idx]

# training data normalization
x_mean, x_std = x_train.mean(), x_train.std()
y_mean, y_std = y_train.mean(), y_train.std()

x_train_norm = (x_train - x_mean) / x_std
x_val_norm   = (x_val   - x_mean) / x_std

y_train_norm = (y_train - y_mean) / y_std
y_val_norm   = (y_val   - y_mean) / y_std

optimizer_sgd = Optimizer_SGD(c.LEARNING_RATE_SGD)
optimizer_sgd_decay = Optimizer_SGD_Decay(c.LEARNING_RATE_SGD_W_DECAY, c.STEP, c.LEARNING_RATE_DECAY)
optimizer_sgd_momentum = Optimizer_SGD_Momentum(c.LEARNING_RATE_MOMENTUM, c.MOMENTUM_BETA)
optimizer_adagrad = Optimizer_Adagrad(c.LEARNING_RATE_ADAGRAD, c.EPSILON)
optimizer_rmsprop = Optimizer_RMSProp(c.LEARNING_RATE_RMSPROP, c.RHO, c.EPSILON)
optimizer_adam = Optimizer_Adam(c.LEARNING_RATE_ADAM, c.ADAM_BETA_1, c.ADAM_BETA_2, c.EPSILON)

layer1 = Layer(c.N_FEATURES, c.N_NEURONS_L1)
layer2 = Layer(c.N_NEURONS_L1, c.N_NEURONS_L2)
layer3 = Layer(c.N_NEURONS_L2, c.N_NEURONS_L3)
layer_output = Layer(c.N_NEURONS_L3, c.N_NEURONS_OUTPUT)

activation1 = Activation_Sigmoid()
activation2 = Activation_Sigmoid()
activation3 = Activation_Sigmoid()
activation_output = Activation_linear()

loss = Loss_MSE()

training_losses = []
validation_losses = []

for i in range(c.N_EPOCHS):

    layer1.forward(x_train_norm)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    layer3.forward(activation2.output)
    activation3.forward(layer3.output)

    layer_output.forward(activation3.output)
    activation_output.forward(layer_output.output)
    y_pred_train = activation_output.output

    training_loss = loss.calculate(y_pred_train, y_train_norm)
    #regularization_training_loss = loss.regularization_loss(layer1) + loss.regularization_loss(layer2) + loss.regularization_loss(layer3)
    total_training_loss = training_loss #+ regularization_training_loss
    training_losses.append(total_training_loss)

    # backward pass of the data
    loss.backward(y_pred_train, y_train_norm)
    activation_output.backward(loss.d_loss)
    layer_output.backward(activation_output.error_signal)

    activation3.backward(activation_output.error_signal, layer_output.weights)
    layer3.backward(activation3.error_signal)

    activation2.backward(activation3.error_signal, layer3.weights)
    layer2.backward(activation2.error_signal)

    activation1.backward(activation2.error_signal, layer2.weights)
    layer1.backward(activation1.error_signal)

    optimizer_sgd.update_parameters(layer1)
    optimizer_sgd.update_parameters(layer2)
    optimizer_sgd.update_parameters(layer3)
    optimizer_sgd.update_parameters(layer_output)

    # data validation
    layer1.forward(x_val_norm)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    layer3.forward(activation2.output)
    activation3.forward(layer3.output)

    layer_output.forward(activation3.output)
    activation_output.forward(layer_output.output)
    y_pred_val = activation_output.output

    validation_loss = loss.calculate(y_pred_val , y_val_norm)
    #regularization_validation_loss = loss.regularization_loss(layer1) + loss.regularization_loss(layer2) + loss.regularization_loss(layer3)
    total_validation_loss = validation_loss #+ regularization_validation_loss
    validation_losses.append(total_validation_loss)

    if i % 2000 == 0:
        print("Epoch: ", i)
        print()
        print("      Training loss: ", total_training_loss)
        print("    Validation loss: ", total_validation_loss)
        print()


# return data to original format
y_pred_train_original = y_pred_train * y_std + y_mean
y_pred_val_original = y_pred_val * y_std + y_mean


traning_data_plot = True
val_data_plot = False
training_loss_plot = False
val_loss_plot = False

if traning_data_plot:
    plt.scatter(x_train, y_train)
    plt.scatter(x_train, y_pred_train_original)

if val_data_plot:
    plt.scatter(x_val, y_val)
    plt.scatter(x_val, y_pred_val_original)

if training_loss_plot:
    plt.plot(range(len(training_losses)), training_losses)
    plt.figure(figsize=(8, 5))
    plt.plot(training_losses, linewidth=2)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log scale)")
    plt.grid(True)


if val_loss_plot:
    plt.plot(range(len(validation_losses)), validation_losses)
    plt.figure(figsize=(8, 5))
    plt.plot(validation_losses, linewidth=2)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log scale)")
    plt.grid(True)

plt.show()

