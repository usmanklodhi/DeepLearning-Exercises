import copy
import math
import numpy as np
import scipy.signal

from .Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.input_tensor = None
        self.trainable = True # Indicates that the layer has trainable parameters

        self.num_kernels = num_kernels
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape

        # Initialize weights and bias for the kernels
        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.random(num_kernels)

        self._padding = "same"
        self._optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None

    def forward(self, input_tensor):
        # Perform the forward pass of the convolution layer
        batch_size = input_tensor.shape[0]
        if len(input_tensor.shape) == 3:
            # For 1D convolution
            hin = input_tensor.shape[2]
            hout = math.ceil(hin / self.stride_shape[0])
            out = np.zeros((batch_size, self.num_kernels, hout))
        if len(input_tensor.shape) == 4:
            # For 2D convolution
            hin = input_tensor.shape[2]
            win = input_tensor.shape[3]
            hout = math.ceil(hin / self.stride_shape[0])
            wout = math.ceil(win / self.stride_shape[1])
            out = np.zeros((batch_size, self.num_kernels, hout, wout))

        self.input_tensor = input_tensor  # storing for backprop

        for item in range(batch_size):
            for ker in range(self.num_kernels):
                # Perform the convolution operation
                output = scipy.signal.correlate(input_tensor[item], self.weights[ker],
                                                self._padding)  # same padding with stride 1 -> input size = output size
                output = output[output.shape[0] // 2]  # valid padding along channels --> drops channels
                if len(self.stride_shape) == 1:
                    output = output[::self.stride_shape[0]]  # stride --> skip
                elif len(self.stride_shape) == 2:
                    output = output[::self.stride_shape[0], ::self.stride_shape[1]]  # diff stride in diff spatial dims
                out[item, ker] = output + self.bias[ker]
        return out

    def backward(self, errT):
        # Perform the backward pass of the convolution layer
        batch_size = np.shape(errT)[0]
        num_channels = self.convolution_shape[0]

        # Prepare weights for convolution
        weights = np.swapaxes(self.weights, 0, 1)
        weights = np.fliplr(weights)
        error_per_item = np.zeros((batch_size, self.num_kernels, *self.input_tensor.shape[2:]))
        dX = np.zeros((batch_size, num_channels, *self.input_tensor.shape[2:]))
        for item in range(batch_size):
            for ch in range(num_channels):
                if len(self.stride_shape) == 1:
                    error_per_item[:, :, ::self.stride_shape[0]] = errT[item]
                else:
                    error_per_item[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = errT[item]
                output = scipy.signal.convolve(error_per_item[item], weights[ch], 'same')
                output = output[output.shape[0] // 2]
                dX[item, ch] = output

        # Compute gradients for weights and bias
        self._gradient_weights, self._gradient_bias = self.get_weights_biases_gradient(errT)
        if self.optimizer is not None:
            # Update weights and bias using optimizer
            self.weights = copy.deepcopy(self.optimizer).calculate_update(self.weights, self._gradient_weights)
            self.bias = copy.deepcopy(self.optimizer).calculate_update(self.bias, self._gradient_bias)

        return dX

    def get_weights_biases_gradient(self, errT):
        # Compute gradients for weights and bias

        global dB
        batch_size = np.shape(errT)[0]
        num_channels = self.convolution_shape[0]
        dW = np.zeros((self.num_kernels, *self.convolution_shape))
        error_per_item = np.zeros((batch_size, self.num_kernels, *self.input_tensor.shape[2:]))
        for item in range(batch_size):
            if len(self.stride_shape) == 1:
                error_per_item[:, :, ::self.stride_shape[0]] = errT[item]
                dB = np.sum(errT, axis=(0, 2))
                padding_width = ((0, 0), (self.convolution_shape[1] // 2, (self.convolution_shape[1] - 1) // 2))
            else:
                error_per_item[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = errT[item]
                dB = np.sum(errT, axis=(0, 2, 3))
                padding_width = ((0, 0), (self.convolution_shape[1] // 2, (self.convolution_shape[1] - 1) // 2),
                                 (self.convolution_shape[2] // 2, (self.convolution_shape[2] - 1) // 2))

            # Pad the input tensor for valid convolution
            paded_X = np.pad(self.input_tensor[item], padding_width, mode='constant', constant_values=0)
            tmp = np.zeros((self.num_kernels, *self.convolution_shape))
            for ker in range(self.num_kernels):
                for ch in range(num_channels):
                    tmp[ker, ch] = scipy.signal.correlate(paded_X[ch], error_per_item[item][ker], 'valid')
            dW += tmp
        return dW, dB

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def initialize(self, weights_initializer, bias_initializer):
        # Initialize weights and bias using the provided initializers
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape),
                                                      self.num_kernels * np.prod(self.convolution_shape[1:]))
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)
