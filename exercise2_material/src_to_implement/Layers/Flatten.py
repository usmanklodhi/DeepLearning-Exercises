import numpy as np


class Flatten:
    def __init__(self):
        self.batch_size = None
        self.trainable = False
        self.width = None
        self.height = None
        self.depth = None

    def forward(self, input_tensor):
        # Perform the forward pass by flattening the input tensor

        # Extract the shape of the input tensor
        batch_size, width, height, depth = np.shape(input_tensor)

        # Store the dimensions for use in the backward pass
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.depth = depth
        # Flatten the input tensor into shape (batch_size, width * height * depth)
        return np.reshape(input_tensor, (batch_size, width * height * depth))

    def backward(self, error_tensor):
        # Perform the backward pass by reshaping the error tensor back to the original shape

        # Reshape the error tensor to the original input shape
        return np.reshape(error_tensor, (self.batch_size, self.width, self.height, self.depth))
