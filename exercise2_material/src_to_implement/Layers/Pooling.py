import numpy as np
import math

from .Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.max_indices = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channels, height, width = input_tensor.shape

        # Calculate output dimensions
        pooled_height = (height - self.pooling_shape[0]) // self.stride_shape[0] + 1
        pooled_width = (width - self.pooling_shape[1]) // self.stride_shape[1] + 1

        # Initialize output tensor
        output_tensor = np.zeros((batch_size, channels, pooled_height, pooled_width))

        # Initialize matrix to score max indices for the backward pass
        self.max_indices = np.zeros_like(output_tensor, dtype=int)

        # Perform pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(pooled_height):
                    for j in range(pooled_width):
                        h_start = i * self.stride_shape[0]
                        w_start = j * self.stride_shape[1]
                        h_end = h_start + self.pooling_shape[0]
                        w_end = w_start + self.pooling_shape[1]

                        pool_region = input_tensor[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(pool_region)
                        output_tensor[b, c, i, j] = max_val

                        # Store the index of the max element
                        max_index = np.argmax(pool_region)
                        self.max_indices[b, c, i, j] = max_index

        return output_tensor

    def backward(self, error_tensor):
        input_gradient = np.zeros_like(self.input_tensor)

        batch_size, channels, pooled_height, pooled_width = error_tensor.shape
        for b in range(batch_size):
            for c in range(channels):
                for i in range(pooled_height):
                    for j in range(pooled_width):
                        # Find the position of the max value stored during forward pass
                        max_index = self.max_indices[b, c, i, j]
                        h_start = i * self.stride_shape[0]
                        w_start = j * self.stride_shape[1]
                        h_index = h_start + max_index // self.pooling_shape[1]
                        w_index = w_start + max_index % self.pooling_shape[1]

                        # Route the gradient to the max position
                        input_gradient[b, c, h_index, w_index] = error_tensor[b, c, i, j]
        return input_gradient

