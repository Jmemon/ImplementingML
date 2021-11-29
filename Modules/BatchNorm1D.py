import numpy as np
from Modules.Module import Module


class BatchNorm1D(Module):

    def __init__(self, size: int, eps: float = 0.001):
        super(BatchNorm1D, self).__init__()

        self.size = size
        self.gamma = np.ones((size, 1))
        self.beta = np.zeros((size, 1))

        self.eps = eps  # small float to avoid division by zero

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2  # height x batch_size
        assert x.shape[0] == self.size

        # x - (1/batch_size)(sum(x) along mini-batch)
        # Make mean for each dimension along mini-batch 0
        z_tilde = x - (np.sum(x, axis=1) / x.shape[1]).reshape(x.shape[0], 1)

        # z_tilde / sqrt(eps + (1/batch_size)(sum( (z_tilde - mu)^2 ) along mini-batch) – mu is zero from prev step
        # Make stdev for each dimension along mini-batch 1
        z_hat = z_tilde / np.sqrt(self.eps + (np.sum(np.square(z_tilde), axis=1) / x.shape[1]).reshape(x.shape[0], 1))

        # gamma * z_hat + self.beta
        # gamma = new stdev –– beta = new mean
        return np.multiply(self.gamma, z_hat) + self.beta

    def backward(self, x: np.ndarray, dy: np.ndarray) -> (np.ndarray, np.ndarray):
        assert x.ndim == 2, "should be (size, batch_size)"
        assert x.shape[0] == self.size, "input size mismatch"

        assert dy.ndim == 2, "should be (size, batch_size)"
        assert dy.shape[0] == self.size, "input size mismatch"

        assert x.shape[1] == dy.shape[1], "different number of batches"

        dl_prev = np.empty(x.shape)
        d_params = list()  # [d_gamma d_beta]

        mu = (np.sum(x, axis=1) / x.shape[1]).reshape(x.shape[0], 1)
        sigma = (np.sum(np.square(x - mu), axis=1) / x.shape[1]).reshape(x.shape[0], 1)

        z = x - mu
        z = z / np.sqrt(self.eps + sigma).reshape(x.shape[0], 1)

        assert z.shape == x.shape

        dg = np.sum(np.multiply(dy, z), axis=1)
        db = np.sum(dy, axis=1)
        d_params = [dg, db]

        dl_dz = np.multiply(dy, self.gamma)
        dz_dx = np.multiply(-1. / np.sqrt(self.eps + sigma).reshape(x.shape[0], 1),
                            (np.multiply(z, z) / 2.) - float(x.shape[1] - 1) / float(x.shape[1]))
        dl_prev = np.multiply(dl_dz, dz_dx)

        return dl_prev, d_params

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
