import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.label_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor
        epsilon = np.finfo(float).eps
        # limit the values in an array to a specified range
        prediction_tensor = np.clip(prediction_tensor, epsilon, 1 - epsilon)
        loss = -np.sum(label_tensor * np.log(prediction_tensor)) / prediction_tensor.shape[0]
        return loss

    def backward(self, label_tensor):
        batch_size = self.prediction_tensor.shape[0]
        return (self.prediction_tensor - label_tensor) / batch_size