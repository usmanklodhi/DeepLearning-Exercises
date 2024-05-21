import numpy as np


class BaseLayer(object):
    def __init__(self):
        self.trainable = False
        # with random values drawn from a uniform distribution,which can provide a good starting point for
        # optimization algorithms like gradient descent.
        self.weights = np.random.uniform(-1, 1, size=(10, 10))
