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
        # Adding 1 to input_size to accommodate bias in the weights' matrix.
        self.weights = np.random.uniform(0, 1, (output_size, input_size + 1))
        self._optimizer = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # Column vector filled with ones with shape (number of samples in input_tensor, 1)
        bias_term = np.ones((input_tensor.shape[0], 1))
        # Horizontally stack each row of input tensor with 1 at the end
        augmented_input = np.hstack([input_tensor, bias_term])
        # Perform matrix multiplication
        return np.dot(augmented_input, self.weights.T)

    def backward(self, error_tensor):
        bias_term = np.ones((self.input_tensor.shape[0], 1))
        augmented_input = np.hstack([self.input_tensor, bias_term])
        self.gradient_weights = np.dot(error_tensor.T, augmented_input)
        if self._optimizer:  # optimizer is set
            sgd = self.optimizer
            self.weights = sgd.calculate_update(self.weights, self.gradient_weights)
        # Return the gradient wrt input (excluding the bias)
        return np.dot(error_tensor, self.weights[:,:-1])

    def initialize(self, weights_initializer, bias_initializer):
        # Initialize the weights (excluding the bias term)
        self.weights[:, :-1] = weights_initializer.initialize((self.output_size, self.input_size), self.input_size,
                                                              self.output_size)
        # Initialize the bias (last column in weights matrix)
        self.weights[:, -1] = bias_initializer.initialize((self.output_size,), self.input_size, self.output_size)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
