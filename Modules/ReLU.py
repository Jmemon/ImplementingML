import numpy as np
from Module import Module


class ReLU(Module):

    def __init__(self, size: int):
        super(ReLU, self).__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.max(0, x)

    def backward(self, a: np.ndarray) -> np.ndarray:
        return np.where(a < 0, 0.0, 1.0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
