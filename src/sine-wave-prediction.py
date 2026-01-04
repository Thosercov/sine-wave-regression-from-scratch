import numpy as np
import matplotlib.pyplot as plt
import constants as c
from layer import Layer


np.random.seed(0)


x_samples = np.random.uniform(low = 0.0, high = 2 * np.pi, size = (c.N_SAMPLES, 1))
y_samples = np.sin(x_samples) + np.random.normal(loc = 0.0, scale = 0.3, size = (c.N_SAMPLES, 1))

#plt.scatter(x_samples, y_samples)
#plt.show()

layer1 = Layer(c.N_SAMPLES, 2)

print(layer1.weights)