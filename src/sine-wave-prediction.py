import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)

n_samples = 200
x_samples = np.random.uniform(low = 0.0, high = 2 * np.pi, size = (n_samples, 1))
y_samples = np.sin(x_samples) + np.random.normal(loc = 0.0, scale = 0.3, size = (n_samples, 1))

plt.scatter(x_samples, y_samples)
plt.show()