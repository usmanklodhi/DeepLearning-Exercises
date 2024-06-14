import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        # Assertion to ensure learning_rate is a float
        # assert isinstance(learning_rate, float)
        # Constructor to initialize the learning rate
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Method to calculate the updated weights using Stochastic Gradient Descent (SGD)

        # The update rule for SGD is: new_weight = old_weight - (learning_rate * gradient)
        # Here, weight_tensor represents the current weights and gradient_tensor represents the gradients

        updated_weights = weight_tensor - self.learning_rate * gradient_tensor

        return updated_weights


class Adam:
    def __init__(self, learning_rate, mu, rho):
        # Initialize the Adam optimizer with the given parameters
        self.learning_rate = learning_rate  # Learning rate for the optimizer
        self.mu = mu  # Decay rate for the first moment estimate (moving average of the gradient)
        self.rho = rho  # Decay rate for the second moment estimate (moving average of the squared gradient)
        self.v = 0  # Initialize first moment vector (mean of gradients)
        self.r = 0  # Initialize second moment vector (mean of squared gradients)
        self.k = 0  # Initialize timestep to keep track of updates

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Compute the Adam update for the given weights and gradients

        gt = gradient_tensor
        mu = self.mu
        rho = self.rho
        self.k += 1

        # Update biased first moment estimate
        self.v = mu * self.v + (1 - mu) * gt
        # Update biased second moment estimate
        self.r = rho * self.r + (1 - rho) * gt * gt

        # Compute bias-corrected first moment estimate
        v_hat = self.v / (1 - np.power(mu, self.k))
        # Compute bias-corrected second moment estimate
        r_hat = self.r / (1 - np.power(rho, self.k))

        # Update weights using the Adam optimization formula
        # Add a small epsilon value to avoid division by zero
        updated_weight = weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + np.finfo(float).eps)
        return updated_weight


class SgdWithMomentum:

    # Initialize the SGD with Momentum optimizer with the given parameters
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Compute the SGD with Momentum update for the given weights and gradients
        # Update the velocity vector
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        # Update the weights using the velocity vector
        updated_weight = weight_tensor + self.v
        return updated_weight
