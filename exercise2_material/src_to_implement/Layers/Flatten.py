import numpy as np


class Flatten:
    def __init__(self):
        # stores the shape of the input tensor, which is used in the backward pass
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        batch_size = self.input_shape[0]
        # -1 in reshape is a special placeholder that infers the size of the second dimension such that the total
        # number of elements remain the same
        tensor_flattened = input_tensor.reshape(batch_size, -1)
        return tensor_flattened

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)
