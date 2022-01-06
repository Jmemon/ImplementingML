import numpy as np
from Modules.Conv import Conv
from Modules.CrossEntropyLoss import CrossEntropyLoss
from Modules.BatchNorm1D import BatchNorm1D
from Modules.RNN import RNN


def main():
    in_size = 100
    out_size = 1
    batch_size = 100

    rnn = RNN(in_size=in_size, out_size=out_size, batch_size=100)
    x = np.random.random((556, in_size, batch_size))

    print(rnn(x))


if __name__ == "__main__":
    main()
