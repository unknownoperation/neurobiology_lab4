import numpy as np
from tqdm import tqdm
import pickle

from snn import SNN, is_target_spiking
from mnist_utils import get_train_data, get_test_data, convert_MNIST_image_to_spikes_in_time


def simulation1():
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


def simulation2(train_X, train_y, test_X, test_y):
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

    lam = 25

    input_layer_size = 5
    T = 100

    input_pattern = generate_input(T, input_layer_size)

    epoch_cnt = 5
    snn = SNN(input_layer_size, 10, 20, 2, u_spike=0.3, T_max=T * epoch_cnt)
    for epoch in range(epoch_cnt):
        for tt in range(T):
            snn.apply(input_pattern[tt], epoch * T + tt)
            snn.STDP(0, 1, 1.7, 2.0, 0.3, 0.2, tt, is_visualize=True)
    snn.visualize_learning()


def simulation3(train_X, train_y, test_X, test_y):
    T = 10
    epoch_cnt = len(train_y)
    snn = SNN(256, 128, 64, 10, u_spike=0.3, T_max=T * epoch_cnt)
    # one epoch -> one image
    for epoch in range(epoch_cnt):
        input_pattern = convert_MNIST_image_to_spikes_in_time(train_X[epoch], T)
        target = int(train_y[epoch])

        print(epoch)
        for tt in tqdm(range(T)):
            cur_time = epoch * T + tt
            snn.apply(input_pattern[tt], cur_time)
            snn.calculate_reward(cur_time, target, 1.2, 10)
            snn.STDP(0, 1, 1.7, 2.0, 0.3, 0.2, cur_time, is_visualize=True)
        snn.reset_reward(epoch * (T + 1) - 1)
        # snn.visualize_reward()

    with open('aaa.pkl', 'wb') as f:
        pickle.dump(snn, f)

    acc_sum = 0
    for test_idx in range(len(test_y)):
        input_pattern = convert_MNIST_image_to_spikes_in_time(test_X[test_idx], T)
        target = int(test_y[test_idx])

        for tt in range(T):
            snn.apply(input_pattern[tt], tt)
            if is_target_spiking(snn.spike_matrix3[tt], target):
                acc_sum += 1
    accuracy = acc_sum / (T * 100)
    print(accuracy)
        

if __name__ == "__main__":
    train_X, train_y = get_train_data()
    print(train_X.shape)
    print(train_y.shape)
    test_X, test_y = get_test_data()
    print(test_X.shape)
    print(test_y.shape)
    # print(convert_MNIST_image_to_spikes_in_time(train_X[0], 10))

    #simulation1()
    #simulation2(train_X, train_y, test_X, test_y)
    simulation3(train_X, train_y, test_X, test_y)

