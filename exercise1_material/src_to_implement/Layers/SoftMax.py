import numpy as np
from .Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.softmax_output = None

    def forward(self, input_tensor):
        # Subtracting maximum value in each row from input tensor
        # to prevent overflow
        # Each element of the adjusted input tensor is exponentiated.
        exp_tensor = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        # Normalization step
        self.softmax_output = exp_tensor / np.sum(exp_tensor, axis=1, keepdims=True)
        return self.softmax_output

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]
        grad_input = self.softmax_output - error_tensor
        return grad_input / batch_size
