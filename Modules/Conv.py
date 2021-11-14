import numpy as np
from Modules.Module import Module


class Conv(Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super(Conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernels = np.random.rand(kernel_size, kernel_size, in_channels, out_channels)
        self.stride = stride

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 4  # (height, width, in_channels, batch_size)
        assert x.shape[2] == self.in_channels

        out = np.zeros((-(x.shape[0] // -self.stride),
                        -(x.shape[1] // -self.stride),
                        self.out_channels,
                        x.shape[3]))

        def apply_filter(c: np.ndarray, k: np.ndarray) -> np.ndarray:
            assert c.ndim == 2  # channel
            assert k.ndim == 2  # kernel

            c_out = np.zeros((-(c.shape[0] // -self.stride), -(c.shape[1] // -self.stride)))

            k_half = self.kernel_size // 2
            pad = np.pad(c, (k_half, k_half - (1 - k_half % 2)))

            for row in range(k_half, c.shape[0] + k_half, self.stride):
                r_slice = slice(row - k_half, row + k_half + (self.kernel_size % 2))

                for col in range(k_half, channel.shape[1] + k_half, self.stride):
                    c_slice = slice(col - k_half, col + k_half + (self.kernel_size % 2))

                    idx = ((row - k_half) // self.stride, (col - k_half) // self.stride)
                    c_out[idx] = np.sum(np.multiply(k, pad[r_slice, c_slice]))

            return c_out

        for b in range(x.shape[3]):  # Go through every element of the batch
            for out_ch in range(self.out_channels):  # Build each channel in a sample's output
                for in_ch in range(self.in_channels):  # Go through each input channel
                    channel = x[:, :, in_ch, b]
                    out[:, :, out_ch, b] = np.add(apply_filter(channel,
                                                               self.kernels[:, :, in_ch, out_ch]),
                                                  out[:, :, out_ch, b])

        return out

    def backward(self, dz: np.ndarray, a: np.ndarray) -> (np.ndarray, np.ndarray):
        assert dz.ndim == 4
        assert dz.shape[2] == self.out_channels

        assert a.ndim == 4
        assert a.shape[2] == self.in_channels

        assert a.shape[3] == dz.shape[3]  # same batch_size

        da = np.zeros(a.shape)
        dw = np.empty(self.kernels.shape)

        def build_pixel_deriv_wrt_kernel(c: np.ndarray) -> np.ndarray:
            assert c.ndim == 2  # channel to take kernel-sized slices from - shape = (height, width)

            # each k x k matrix is a derivative of one pixel wrt kernel, so tensor is the derivative for all pixels
            delta = np.zeros((self.kernel_size, self.kernel_size, dz.shape[0] * dz.shape[1]))

            k_half = self.kernel_size // 2
            pad = np.pad(c, (k_half, k_half - (1 - k_half % 2)))

            for row in range(k_half, k_half + c.shape[0], self.stride):
                r_slice = slice(row - k_half, row + k_half + (self.kernel_size % 2))

                for col in range(k_half, k_half + c.shape[1], self.stride):
                    c_slice = slice(col - k_half, col + k_half + (self.kernel_size % 2))

                    idx = ((row - k_half) // self.stride) * dz.shape[1] + (col - k_half) // self.stride
                    delta[:, :, idx] = pad[r_slice, c_slice]

            return delta

        def build_pixel_deriv_wrt_a(kernel: np.ndarray) -> np.ndarray:
            assert kernel.ndim == 2

            delta = np.zeros((a.shape[0], a.shape[1], dz.shape[0] * dz.shape[1]))

            k_half = self.kernel_size // 2

            for p_out in range(delta.shape[2]):
                a_row = (p_out // dz.shape[1]) * self.stride
                a_col = (p_out % dz.shape[1]) * self.stride

                parity_off = (self.kernel_size & 1) - 1

                delta_row_slice = slice(a_row - k_half if a_row - k_half >= 0 else 0,
                                        a_row + k_half + parity_off + 1 if a_row + k_half + parity_off + 1 <= a.shape[0]
                                        else a.shape[0])
                delta_col_slice = slice(a_col - k_half if a_col - k_half >= 0 else 0,
                                        a_col + k_half + parity_off + 1 if a_col + k_half + parity_off + 1 <= a.shape[1]
                                        else a.shape[1])

                kernel_row_slice = slice(0 if k_half - a_row < 0 else k_half - a_row,
                                         self.kernel_size if a_row + k_half + parity_off < a.shape[0]
                                         else self.kernel_size - (a.shape[0] - a_row))
                kernel_col_slice = slice(0 if k_half - a_col < 0 else k_half - a_col,
                                         self.kernel_size if a_col + k_half + parity_off < a.shape[1]
                                         else self.kernel_size - (a.shape[1] - a_col))

                assert delta_row_slice.stop - delta_row_slice.start == kernel_row_slice.stop - kernel_row_slice.start
                assert delta_col_slice.stop - delta_col_slice.start == kernel_col_slice.stop - kernel_col_slice.start

                delta[delta_row_slice, delta_col_slice, p_out] = kernel[kernel_row_slice, kernel_col_slice]

            return delta

        for b in range(a.shape[3]):
            for out_ch in range(self.out_channels):
                for in_ch in range(self.in_channels):
                    channel = a[:, :, in_ch, b]
                    kernel = self.kernels[:, :, in_ch, out_ch]
                    
                    p_deltas_wrt_kernel = build_pixel_deriv_wrt_kernel(channel)  # k_size x k_size x dz_h * dz_w
                    p_deltas_wrt_a = build_pixel_deriv_wrt_a(kernel)  # a_h x a_w x dz_h * dz_w
                    scalars = dz.reshape(-1, dz.shape[2], dz.shape[3])[:, out_ch, b]  # dz_h * dz_w

                    assert scalars.ndim == 1
                    assert p_deltas_wrt_kernel.ndim == 3
                    assert p_deltas_wrt_a.ndim == 3

                    assert scalars.shape == (channel.shape[0] * channel.shape[1],)
                    assert scalars.shape == (dz.shape[0] * dz.shape[1],)
                    assert p_deltas_wrt_kernel.shape == (self.kernel_size,
                                                         self.kernel_size,
                                                         channel.shape[0] * channel.shape[1])
                    assert p_deltas_wrt_a.shape == (a.shape[0], a.shape[1], dz.shape[0] * dz.shape[1])

                    assert p_deltas_wrt_kernel.shape[2] == scalars.shape[0]
                    assert p_deltas_wrt_a.shape[2] == scalars.shape[0]

                    da[:, :, in_ch, b] += np.sum(np.multiply(scalars, p_deltas_wrt_a), axis=2)
                    dw[:, :, in_ch, out_ch] = np.sum(np.multiply(scalars, p_deltas_wrt_kernel), axis=2)

        return da, dw

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
