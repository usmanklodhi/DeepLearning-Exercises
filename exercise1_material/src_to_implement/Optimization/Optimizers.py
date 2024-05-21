import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        # Assertion to ensure learning_rate is a float
        assert isinstance(learning_rate, float)
        # Constructor to initialize the learning rate
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Method to calculate the updated weights using Stochastic Gradient Descent (SGD)

        # The update rule for SGD is: new_weight = old_weight - (learning_rate * gradient)
        # Here, weight_tensor represents the current weights and gradient_tensor represents the gradients

        updated_weights = weight_tensor - self.learning_rate * gradient_tensor

        return updated_weights
