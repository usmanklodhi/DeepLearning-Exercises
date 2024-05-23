import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.gradient_biases = None
        self.gradient_weights = None
        self.input_tensor = None
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (output_size, input_size))
        self.biases = np.random.uniform(0, 1, (output_size, 1))
        self._optimizer = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.dot(self.weights, input_tensor.T).T + self.biases.T

    def backward(self, error_tensor):
        self.gradient_weights = np.dot(error_tensor.T, self.input_tensor)  # Gradient wrt weights
        self.gradient_biases = np.sum(error_tensor.T, axis=1, keepdims=True)  # Gradient wrt biases
        if self._optimizer:  # optimizer is set
            sgd = self.optimizer
            self.weights = sgd.calculate_update(self.weights, self.gradient_weights)
            self.biases = sgd.calculate_update(self.biases, self.gradient_biases)
        return np.dot(error_tensor, self.weights)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
