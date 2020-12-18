import numpy as np

from snn import SNN


if __name__ == "__main__":
    lam = 25

    input_layer_size = 30
    T = 100

    input_pattern = np.zeros((T, input_layer_size))
    for i in range(input_layer_size):
        j = -1
        while j < T:
            j += np.random.poisson(lam, 1)
            if j >= T:
                break
            input_pattern[j, i] = 1

    print(input_pattern)

    snn1 = SNN(input_layer_size, 40, 50, 2)

    epoch_cnt = 5
    for epoch in range(epoch_cnt):
        for tt in range(T):
            snn1.apply(input_pattern[tt], epoch * T + tt)

    snn1.visualize_u(T * epoch_cnt)
