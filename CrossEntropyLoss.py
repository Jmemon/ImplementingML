import numpy as np
from Module import Module


class CrossEntropyLoss(Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, model_out: np.ndarray, actual: np.ndarray) -> float:
        assert model_out.ndim == 2  # classes x batch_size
        assert actual.ndim == 1  # batch_size

        assert model_out.shape[1] == actual.shape[0]

        one_hot_actual = np.zeros(model_out.shape)

        for i in range(actual.shape[0]):
            one_hot_actual[actual[i]][i] = 1

        return -np.sum(np.multiply(np.log2(model_out), one_hot_actual))

    def backward(self, model_out: np.ndarray, actual: np.ndarray) -> np.ndarray:
        assert model_out.ndim == 2  # classes x batch_size
        assert actual.ndim == 1  # batch_size

        assert model_out.shape[1] == actual.shape[0]

        one_hot_actual = np.zeros(model_out.shape)

        for i in range(actual.shape[0]):
            one_hot_actual[actual[i]][i] = 1

        return -np.sum(np.multiply(np.reciprocal(np.log(2) * model_out), one_hot_actual))
