import numpy as np
from .Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.bias = None
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
        return np.dot(error_tensor, self.weights[:, :-1])

    # Uses provided weight and bias initializers to set the initial weights of the layer.

    def initialize(self, weights_initializer, bias_initializer):
        weights_shape = (self.input_size, self.output_size)
        bias_shape = (1, self.output_size)
        # Define fan-in and fan-out for initialization purposes
        fan_in = self.input_size
        fan_out = self.output_size
        # Initialize weights using the provided initializer
        self.weights = weights_initializer.initialize(weights_shape, fan_in, fan_out)
        # Initialize biases using the provided initializer
        self.bias = bias_initializer.initialize(bias_shape, fan_in, fan_out)
        # Concatenate weights and biases along the axis 0
        self.weights = np.concatenate([self.weights, self.bias], axis=0)


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
