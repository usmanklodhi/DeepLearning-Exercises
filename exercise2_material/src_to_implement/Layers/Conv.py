import math

import numpy as np
import scipy.signal
from scipy.signal import convolve, correlate
from .Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.input_tensor = None
        self.trainable = True

        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        # Initialize weights and biases UNIFORMLY in the range [0, 1)
        self.weights = np.random.uniform(0, 1, (num_kernels,) + convolution_shape)
        self.bias = np.random.uniform(0, 1, (num_kernels,))

        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)

        self._optimizer = None
        # Variable for padding has not been set thus far

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        if len(input_tensor.shape) == 3:
            height_input = input_tensor.shape[2]
            height_output = math.ceil(height_input/self.stride_shape[0])
            output_tensor = np.zeros(batch_size, self.num_kernels, height_output)
        if len(input_tensor.shape) == 4:
            height_input = input_tensor.shape[2]
            width_input = input_tensor.shape[3]
            height_output = math.ceil(height_input/self.stride_shape[0])
            width_output = math.ceil(width_input/self.stride_shape[1])
            output_tensor = np.zeros((batch_size, self.num_kernels, height_output, width_output))

        # Store input parameter for back-propagation
        self.input_tensor = input_tensor

        for i in range(batch_size):
            for k in range(self.num_kernels):
                # output is a temporary variable, used to store results of convolution operations
                output = scipy.signal.correlate(input_tensor[i], self.weights[k], "same")
                output = output[output.shape[0] // 2]  # reduce dimensionality of the output from convolution
                if len(self.stride_shape) == 1:
                    output = output[::self.stride_shape[0]]
                elif len(self.stride_shape) == 2:
                    output = output[::self.stride_shape[0], ::self.stride_shape[1]]
                output_tensor[i, k] = output + self.bias[k]
        return output_tensor

    def backward(self, error_tensor):
        batch_size = np.shape(error_tensor)[0]
        num_channels = self.convolution_shape[0]

        # Prepare modified weights for gradient computation
        flipped_weights = np.swapaxes(self.weights, 0, 1)
        flipped_weights = np.fliplr(flipped_weights)

        # Initialize gradients and error arrays
        input_gradient = np.zeros((batch_size, num_channels, *self.input_tensor.shape[2:]))
        weight_gradient = np.zeros((self.num_kernels, *self.convolution_shape))
        bias_gradient = np.zeros(self.num_kernels)

        for batch_index in range(batch_size):
            # Adjust error tensor according to strides for gradient computation
            adjusted_error_tensor = np.zeros((self.num_kernels, *self.input_tensor.shape[2:]))
            if len(self.stride_shape) == 1:
                adjusted_error_tensor[:, ::self.stride_shape[0]] = error_tensor[batch_index]
            else:
                adjusted_error_tensor[:, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[batch_index]

            # Compute gradients for input (input_gradient)
            for channel_index in range(num_channels):
                convoluted_output = scipy.signal.convolve(adjusted_error_tensor, flipped_weights[channel_index], 'same')
                convoluted_output = convoluted_output[convoluted_output.shape[0] // 2]
                input_gradient[batch_index, channel_index] = convoluted_output

            # Compute gradients for weights (weight_gradient) and biases (bias_gradient)
            padding_config = tuple(
                (0, 0) if dim != 1 else (self.convolution_shape[dim] // 2, (self.convolution_shape[dim] - 1) // 2)
                for dim in range(len(self.convolution_shape))
            )
            padded_input = np.pad(self.input_tensor[batch_index], padding_config, mode='constant', constant_values=0)
            for kernel_index in range(self.num_kernels):
                for channel_index in range(num_channels):
                    weight_gradient[kernel_index, channel_index] += scipy.signal.correlate(padded_input[channel_index],
                                                                                           adjusted_error_tensor[
                                                                                               kernel_index], 'valid')

        # Update weights and biases if optimizer is set
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, weight_gradient)
            self.bias = self._optimizer.calculate_update(self.bias, bias_gradient)

        return input_gradient

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @ optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape),
                                                      self.num_kernels * np.prod(self.convolution_shape[1:]))
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)