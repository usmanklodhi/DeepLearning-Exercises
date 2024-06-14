import numpy as np


class Constant:
    def __init__(self, value=0.1):
        """

        :type value: object
        """
        self.value = value

    # Initialize weights to a constant value, typically used for bias terms
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.ones(fan_out) * self.value


class UniformRandom:
    # Initialize weights with a uniform random distribution
    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        return np.random.uniform(size=weights_shape)


class Xavier:
    # Initialize weights using Xavier initialization
    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_in + fan_out))  # Calculate standard deviation
        # Return normally distributed weights with mean 0 and calculated sigma
        return np.random.normal(0, sigma, weights_shape)


class He:
    # Initialize weights using He initialization
    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)  # Calculate standard deviation
        # Return normally distributed weights with mean 0 and calculated sigma
        return np.random.normal(0, sigma, weights_shape)
