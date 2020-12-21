import numpy as np
import pandas as pd

MNIST_IMG_SIZE = 256


def get_train_data():
    train_df = pd.read_csv("mnist_data/mnist_train.psv", sep="|", header=None)
    train_arr = train_df.to_numpy()
    return train_arr[:, 1:], train_arr[:, 0]


def get_test_data():
    test_df = pd.read_csv("mnist_data/mnist_test.psv", sep="|", header=None)
    test_arr = test_df.to_numpy()
    return test_arr[:, 1:], test_arr[:, 0]


def convert_MNIST_image_to_spikes_in_time(data, T):
    assert len(data) == MNIST_IMG_SIZE

    res = np.zeros((T, MNIST_IMG_SIZE))
    for i in range(MNIST_IMG_SIZE):
        pixel_intensity = data[i]
        lam = 1 + T * (1 - (pixel_intensity + 1) / 2.0)

        j = -1
        while j < T:
            j += np.random.poisson(lam, 1)
            if j >= T:
                break
            res[j, i] = 1

    return res
