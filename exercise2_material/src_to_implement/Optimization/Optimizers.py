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


class SgdWithMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            # np.zeros_like returns an array of zeros with the same shape and type as the given array
            self.velocity = np.zeros_like(weight_tensor)
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient_tensor
        return weight_tensor + self.velocity


class Adam:
    def __init__(self, learning_rate=0.001, mu=0.9, rho=0.999):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        # small constant introduced for numerical stability
        self.epsilon = 1e-8
        self.m = None  # first moment estimate: exponentially decaying average of past gradients
        self.v = None  # second moment estimate: exponentially decaying average of past squared gradients
        self.t = 0  # time step

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.m is None:
            self.m = np.zeros_like(weight_tensor)
            self.v = np.zeros_like(weight_tensor)
        self.t += 1
        self.m = self.mu * self.m + (1 - self.mu) * gradient_tensor
        self.v = self.rho * self.v + (1 - self.rho) * (gradient_tensor ** 2)
        #  bias-corrected first and second moment estimates
        m_hat = self.m / (1 - self.mu ** self.t)
        v_hat = self.v / (1 - self.rho ** self.t)
        return weight_tensor - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
