import numpy as np
from Conv import Conv
from CrossEntropyLoss import CrossEntropyLoss


def main():
    x = np.random.rand(5, 5, 3, 1)  # h w c b
    layer = Conv(3, 8, 4, 1)  # in_ch out_ch kernel_size batch_size
    x = layer.forward(x)
    print(x.shape)

    a = np.random.rand(5, 5, 3, 1)
    dz = np.random.rand(5, 5, 8, 1)
    da, dw = layer.backward(dz, a)
    print(da.shape)
    print(dw.shape)

    loss = CrossEntropyLoss()
    out = np.random.rand(10, 4)  # 10 classes, batch_size 4
    print(loss.backward(out, np.array([3, 6, 1, 8])))


if __name__ == "__main__":
    main()
