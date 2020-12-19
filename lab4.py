import numpy as np

from snn import SNN


def generate_input(T, input_layer_size):
    input_pattern = np.zeros((T, input_layer_size))
    for i in range(input_layer_size):
        j = -1
        while j < T:
            j += np.random.poisson(lam, 1)
            if j >= T:
                break
            input_pattern[j, i] = 1
    return input_pattern


if __name__ == "__main__":
    lam = 25

    input_layer_size = 5
    T = 100

    input_pattern = generate_input(T, input_layer_size)

    epoch_cnt = 5
    snn1 = SNN(input_layer_size, 10, 20, 2, u_spike=0.3, T_max=T * epoch_cnt)
    for epoch in range(epoch_cnt):
        for tt in range(T):
            snn1.apply(input_pattern[tt], epoch * T + tt)

    snn1.visualize_most_spiking_u()
