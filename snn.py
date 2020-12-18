import numpy as np
import matplotlib.pyplot as plt


def make_plot(xs, ys, color, label, xlabel, ylabel):
    plt.plot(xs, ys, color, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')


nu = 1
tau = 1
tau_s = 1
u_spike = 1


def eps(s, d_ij):
    return (s - d_ij) / tau_s * np.exp(-(s - d_ij) / tau_s) * (1 if s - d_ij > 0 else 0)


def eta(s):
    return -nu * np.exp(-s / tau) * (1 if s > 0 else 0)


class SNN:
    def __init__(self, input_layer_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_layer_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        self.synapses_cnt1 = np.random.randint(low=1, high=6, size=(input_layer_size, hidden_size1))
        self.synapses_cnt2 = np.random.randint(low=1, high=6, size=(hidden_size1, hidden_size2))
        self.synapses_cnt3 = np.random.randint(low=1, high=6, size=(hidden_size2, output_size))

        self.d1 = np.random.randint(low=1, high=8, size=(input_layer_size, hidden_size1))
        self.d2 = np.random.randint(low=1, high=8, size=(hidden_size1, hidden_size2))
        self.d3 = np.random.randint(low=1, high=8, size=(hidden_size2, output_size))

        self.w1 = np.random.random((input_layer_size, hidden_size1))
        self.w2 = np.random.random((hidden_size1, hidden_size2))
        self.w3 = np.random.random((hidden_size2, output_size))

        self.u1 = np.zeros((1000, hidden_size1))
        self.u2 = np.zeros((1000, hidden_size2))
        self.u3 = np.zeros((1000, output_size))

        self.prev_t_0 = np.zeros(input_layer_size)
        self.prev_t_1 = np.zeros(hidden_size1)
        self.prev_t_2 = np.zeros(hidden_size2)
        self.prev_t_3 = np.zeros(output_size)

    def apply(self, input_layer, t):
        def apply_inner_layer(input_size, output_size, prev_t_n, prev_t_n_1, u, w, d):
            for i in range(output_size):
                u[t, i] = eta(t - prev_t_n_1[i])
                for j in range(input_size):
                    u[t, i] += w[j, i] * eps(t - prev_t_n[j] - d[j, i], d[j, i])
                if u[t, i] > u_spike:
                    prev_t_n_1[i] = t

        for i in range(self.input_size):
            if input_layer[i] == 1:
                self.prev_t_0[i] = t

        apply_inner_layer(self.input_size, self.hidden_size1,
                          self.prev_t_0, self.prev_t_1,
                          self.u1, self.w1, self.d1)

        apply_inner_layer(self.hidden_size1, self.hidden_size2,
                          self.prev_t_1, self.prev_t_2,
                          self.u2, self.w2, self.d2)

        apply_inner_layer(self.hidden_size2, self.output_size,
                          self.prev_t_2, self.prev_t_3,
                          self.u3, self.w3, self.d3)

    def visualize_u(self, maxT):
        tim = np.arange(maxT)
        for i in range(self.output_size):
            make_plot(tim, self.u3[:maxT, i], 'g', 'u(t)', 'time', 'u')
            plt.show()
            plt.close()
