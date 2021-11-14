import numpy as np
from Modules.Module import Module


class Linear(Module):

    def __init__(self, in_size: int, out_size: int):
        super(Linear, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.parameters = np.random.rand(out_size, in_size)  # uniform distribution [0, 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2  # x.shape == (in_size, batch_size)
        assert x.shape[0] == self.in_size

        return np.dot(self.parameters, x)

    def backward(self, dz: np.ndarray, a: np.ndarray) -> (np.ndarray, np.ndarray):
        assert dz.ndim == 2  # dz.shape == (out_size, batch_size)
        assert dz.shape[0] == self.out_size

        assert a.ndim == 2   # a.shape == (in_size, batch_size)
        assert a.shape[0] == self.in_size

        assert a.shape[1] == dz.shape[1]  # same number of batches

        return np.dot(self.parameters.T, dz), np.dot(dz, a.T)   # dL/da, dL/dW

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
