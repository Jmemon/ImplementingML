import math
import numpy as np
from Module import Module


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

        da = np.empty(a.shape)
        dw = np.empty(self.kernels.shape)

        def build_pixel_deriv_wrt_kernel(c: np.ndarray):
            assert c.ndim == 2  # channel to take kernel-sized slices from - shape = (height, width)

            # each k x k matrix is a derivative of one pixel wrt kernel, so tensor is the deriv for all pixels
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

        for b in range(a.shape[3]):
            for out_ch in range(self.out_channels):
                for in_ch in range(self.in_channels):
                    channel = a[:, :, in_ch, b]
                    
                    p_deltas = build_pixel_deriv_wrt_kernel(channel)  # k_size x k_size x h * w
                    scalars = dz.reshape(-1, dz.shape[2], dz.shape[3])[:, out_ch, b]  # h * w

                    assert scalars.ndim == 1
                    assert p_deltas.ndim == 3

                    assert p_deltas.shape == (self.kernel_size, self.kernel_size, channel.shape[0] * channel.shape[1])
                    assert scalars.shape == (channel.shape[0] * channel.shape[1],)

                    assert p_deltas.shape[2] == scalars.shape[0]

                    dw[:, :, in_ch, out_ch] = np.sum(np.multiply(scalars, p_deltas), axis=2)

        return da, dw

    """
    def __init__ (self, input_shape, output_shape, filter_shape, stride):

        # Check Types
        assert isinstance(input_shape, tuple) and len(input_shape) == 4
        assert isinstance(output_shape, tuple) and len(output_shape) == 4
        assert isinstance(filter_shape, tuple) and len(filter_shape) == 3
        assert isinstance(stride, int)

        # Check Shapes
        assert input_shape[1] * filter_shape[0] == output_shape[1]
        assert output_shape[2] * filter_shape[1] == input_shape[2]
        assert output_shape[3] * filter_shape[2] == input_shape[3] 

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.filter_shape = filter_shape
        self.filters = np.random.random(self.filter_shape) - 0.5
        self.stride = int(stride)
         
    def forwardPropagation(self, input_data):
        self.input_data = input_data
        self.output_data = np.zeros(self.output_shape)
        
        # Check Type
        assert isinstance(input_data, np.ndarray) 
        assert self.input_data.shape == self.input_shape
        
        # Check Shape
        assert (self.input_shape[2] - self.filter_shape[1]) % self.stride == 0
        assert (self.input_shape[3] - self.filter_shape[2]) % self.stride == 0
        
        # Filter moves at lower right corner, which is why we start R/C at self.filter_shape[0 or 1] - 1
        # End loops at (in_shape[0 or 1] - filter_shape[0 or 1]) / stride
            # numerator gives distance an entry of the filter will move over the course of multiplication
            # result gives how many iterations it will take to get through that distance, given the stride
        for sample in range(self.input_shape[0]):
            for j in range(self.input_shape[1] * self.filter_shape[0]):
                i_idx = int(np.floor(j / self.filter_shape[0]))
                f_idx = j % self.filter_shape[0]

                row_its = int((self.input_shape[2] - self.filter_shape[1]) / self.stride)
                col_its = int((self.input_shape[3] - self.filter_shape[2]) / self.stride)
                for k in range(row_its * col_its):
                    r = int(np.floor(k / col_its)) * self.stride
                    R = r + self.filter_shape[1]
                    c = (k % col_its) * self.stride
                    C = c + self.filter_shape[2]
               
                    assert self.filters[f_idx].shape == self.input_data[sample, i_idx, r:R, c:C].shape
                    sliceProduct = self.filters[f_idx] * self.input_data[sample, i_idx, r:R, c:C]
                    self.output_data[sample, j, int(r / row_its), int(c / col_its)] = np.sum(sliceProduct)
                
        return self.output_data
    
    # needs to be updated to use minibatch SGD
    def backPropagation(self, delta, alpha):
        Parameters:
            delta - dJ/dz for some layer L (matrix of values) - size is self.output_size
            alpha - learning rate for weight update

        Updates filter weights using SGD 

        Returns:
            dJ/da for layer (L-1) (matrix of values) - size is self.input_size
        
        # Check Types
        assert isinstance(delta, np.ndarray) and delta.ndim == 4
        assert isinstance(alpha, float)
    
        # Check Shapes
        assert self.input_shape[0] == delta.shape[0]
        assert self.input_shape[1] == int(delta.shape[1] / self.filter_shape[0])
        assert self.input_shape[2] == delta.shape[2] * self.filter_shape[1]
        assert self.input_shape[3] == delta.shape[3] * self.filter_shape[2]
        
        delta_prev = np.zeros(self.input_shape)
        
        # Traverse backward
        for i in range(self.output_shape[0] * self.output_shape[1]):
            
            n_idx = int(np.floor(i / self.output_shape[1]))
            i_idx = i % self.output_shape[1]
            f_idx = i % self.filter_shape[0]
            iprev_idx = int(i_idx / self.filter_shape[0])
            
            delta_prev[n_idx, iprev_idx] += matScalarConcatenate(delta[n_idx, i_idx], self.filters[f_idx])
            
            # Update filter
            for j in range(self.output_shape[2] * self.output_shape[3]):
                
                r = int(np.floor(j / self.output_shape[3]))
                c = j % self.output_shape[3]
                
                self.filters[f_idx] -= alpha * (delta[n_idx,i,r,c] * self.filters[f_idx])

        return delta_prev
        
    def printWeights(self):
        Prints each filter 
        
        for f in range(self.filter_shape[0]):
            print("Filter {}:\n".format(f))
            print(self.filters[f])
            print("\n\n")
    
    """
