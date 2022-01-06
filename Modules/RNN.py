import numpy as np

from Modules.Module import Module
from Modules.ReLU import ReLU


class RNN(Module):

    in_size: int
    out_size: int
    batch_size: int

    activation: Module

    parameters_ax: np.ndarray
    parameters_aa: np.ndarray
    bias: np.ndarray

    def __init__(self, in_size: int, out_size: int, batch_size: int = 100, activation: Module = ReLU):
        super(RNN, self).__init__()
        
        self.in_size = in_size
        self.out_size = out_size
        self.batch_size = batch_size
        
        self.activation = activation

        self.parameters_ax = np.random.rand(self.out_size, self.in_size)  # uniform distribution [0, 1)
        self.parameters_aa = np.random.rand(self.out_size, self.out_size)
        self.bias = np.random.rand(self.out_size, 1)

    def check_forward_inputs(self, in_seq: np.ndarray, a_0: np.ndarray) -> None:
        assert len(in_seq.shape) == 3
        assert in_seq.shape[1] == self.in_size
        assert in_seq.shape[2] == self.batch_size

        assert len(a_0.shape) == 2
        assert a_0.shape[0] == self.out_size
        assert a_0.shape[1] == self.batch_size

    def forward(self, in_seq: np.ndarray, a_0: np.ndarray) -> (np.ndarray, list):

        """
        W_ax: H_OUT x H_IN
        in_seq: L x H_IN x N
        W_aa is H_OUT x H_OUT
        a_0: H_OUT x N
        bias: H_OUT x 1
        out: L x H_OUT x N
        where L is the seq length, N is the batch_size, H_IN is the input height, H_OUT is the output height

        Computes:
            a_t = g(W_ax(x_t) + W_aa(a_t-1) + b_a) for t in [0, L-1]
        """

        # matmul (stack of matrices case): https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
        # transpose: https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
        # broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html

        self.check_forward_inputs(in_seq, a_0)

        cache = [None] * 3  # x g' a
        cache[0] = in_seq  # x
        cache[1] = np.empty((in_seq.shape[0], self.out_size, self.batch_size))  # g'
        cache[2] = np.empty((in_seq.shape[0], self.out_size, self.batch_size))  # a

        out = np.matmul(self.parameters_ax, in_seq)

        out[0, :, :] += np.matmul(self.parameters_aa, a_0) + self.bias
        cache[1][0, :, :] = self.activation.backward(out[0, :, :]).copy()

        out[0, :, :] = self.activation(out[0, :, :])
        cache[2][0, :, :] = out[0, :, :].copy()

        for i in range(1, out.shape[0]):  # goes through elements of sequence
            # (H_OUT x H_OUT) * (H_OUT x N) + (H_OUT x N) <-- bias gets broadcast to this

            out[i, :, :] += np.matmul(self.parameters_aa, out[i - 1, :, :]) + self.bias
            cache[1][i, :, :] = self.activation.backward(out[i, :, :]).copy()

            out[i, :, :] = self.activation(out[i, :, :])
            cache[2][i, :, :] = out[i, :, :].copy()

        return out, cache

    def backward(self, delta: np.ndarray, cache: list) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):

        """
        delta: H_OUT x N
        x = cache[0] : L x H_IN  x N
        g' = cache[1]: L x H_OUT x N
        a = cache[2] : L x H_OUT X N

        Computes:
            dL/dW_aa = sum over t (delta_t hadamard g'_t)x_t^T
            dL/dW_ax = sum over t (delta_t hadamard g'_t)a_t-1^T
            dL/db_a = sum over t (delta_t hadamard g'_t)
            dL/dx_t = W_ax^T(delta_t hadamard g'_t)

        Returns:
            dX, dW_ax, dW_aa, db_a
        """

        assert len(delta.shape) == 2
        assert delta.shape[0] == self.out_size
        assert delta.shape[1] == self.batch_size

        assert len(cache) == 3

        assert len(cache[0].shape) == 3
        assert cache[0].shape[1] == self.in_size
        assert cache[0].shape[2] == self.batch_size

        assert len(cache[1].shape) == 3
        assert cache[1].shape[1] == self.out_size
        assert cache[1].shape[2] == self.batch_size

        assert len(cache[2].shape) == 3
        assert cache[2].shape[1] == self.out_size
        assert cache[2].shape[2] == self.batch_size

        delta = np.reshape(delta, (1, delta.shape[0], delta.shape[1]))
        x, g_prime, a = cache

        core = np.multiply(delta, g_prime)  # L x H_OUT x N

        # L x (H_IN x H_OUT * H_OUT x N) --> L x H_IN x N
        dx = np.matmul(self.parameters_ax.transpose((1, 0)), core)

        # L x H_OUT x N * L x N x H_IN -> L x H_OUT x H_IN
        dw_ax = np.matmul(core, x.transpose(0, 2, 1))

        # L x H_OUT x N * L x N x H_OUT -> L x H_OUT x H_OUT
        dw_aa = np.matmul(core, a.transpose(0, 2, 1))

        # db_a == core

        return dx, dw_ax, dw_aa, core

    def __call__(self, in_seq: np.ndarray, a_0: np.ndarray) -> np.ndarray:
        self.check_forward_inputs(in_seq, a_0)
        return self.forward(in_seq, a_0)
