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
        # Gradient of the loss with respect to the input logits
        gradient_logits = self.softmax_output * error_tensor
        # Sum of the gradients across the classes (for each sample in the batch)
        sum_gradients = gradient_logits.sum(axis=1, keepdims=True)
        # Subtract the scaled softmax output from the gradient to adjust for the interaction between classes
        gradient_logits -= self.softmax_output * sum_gradients
        return gradient_logits
