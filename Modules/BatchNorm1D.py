import numpy as np
from Modules.Module import Module


class BatchNorm1D(Module):

    def __init__(self, size: int):
        super(BatchNorm1D, self).__init__()

        self.size = size
        self.gamma = np.ones((size, 1))
        self.beta = np.zeros((size, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2  # height x batch_size
        assert x.shape[0] == self.size

        eps = 0.001  # small float to avoid dividing by zero

        # x - (1/batch_size)(sum(x) along mini-batch)
        # Make mean for each dimension along mini-batch 0
        z_tilde = x - (np.sum(x, axis=1) / x.shape[1]).reshape(x.shape[0], 1)

        # z_tilde / sqrt(eps + (1/batch_size)(sum( (z_tilde - mu)^2 ) along mini-batch) – mu is zero from prev step
        # Make stdev for each dimension along mini-batch 1
        z_hat = z_tilde / np.sqrt(eps + np.sum(np.square(z_tilde), axis=1).reshape(x.shape[0], 1))

        # gamma * z_hat + self.beta
        # gamma = new stdev –– beta = new mean
        return np.multiply(self.gamma, z_hat) + self.beta

    def backward(self, a: np.ndarray, dl: np.ndarray) -> (np.ndarray, np.ndarray):
        assert a.ndim == 2, "should be (size, batch_size)"
        assert a.shape[0] == self.size, "input size mismatch"

        assert dl.ndim == 2, "should be (size, batch_size)"
        assert dl.shape[0] == self.size, "input size mismatch"

        assert a.shape[1] == dl.shape[1], "different number of batches"

        dl_prev = np.empty(a.shape)
        d_params = np.empty((self.size, 2))  # [d_gamma d_beta]

        

        return dl_prev, d_params

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)