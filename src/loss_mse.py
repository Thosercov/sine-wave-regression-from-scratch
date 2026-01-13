import numpy as np

class Loss_MSE:

    def __init__(self, y_prediction, y_true):
        self.y_pred = y_prediction
        self.y_true = y_true

    def forward(self):
        delta = self.y_pred - self.y_true
        self.output = 0.5 * np.mean(delta ** 2)
    
    def backward(self):
        delta = self.y_pred - self.y_true
        n = delta.size
        self.d_loss= delta / n