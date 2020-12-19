import numpy as np
import matplotlib.pyplot as plt


def make_plot(ax, xs, ys, color, label, xlabel, ylabel):
    ax.plot(xs, ys, color, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')


nu = 1
tau = 1
tau_s = 1


def eps(s, d_ij):
    return (s - d_ij) / tau_s * np.exp(-(s - d_ij) / tau_s) * (1 if s - d_ij > 0 else 0)


def eta(s):
    return -nu * np.exp(-s / tau) * (1 if s > 0 else 0)


class SNN:
    def __init__(self, input_layer_size, hidden_size1, hidden_size2, output_size, u_spike=0.3, T_max=1000):
        self.input_size = input_layer_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        self.synapses_cnt1 = np.random.randint(low=1, high=5, size=(input_layer_size, hidden_size1))
        self.synapses_cnt2 = np.random.randint(low=1, high=5, size=(hidden_size1, hidden_size2))
        self.synapses_cnt3 = np.random.randint(low=1, high=5, size=(hidden_size2, output_size))

        self.d1 = np.random.randint(low=1, high=7, size=(input_layer_size, hidden_size1))
        self.d2 = np.random.randint(low=1, high=7, size=(hidden_size1, hidden_size2))
        self.d3 = np.random.randint(low=1, high=7, size=(hidden_size2, output_size))

        self.w1 = np.random.random((input_layer_size, hidden_size1))
        self.w2 = np.random.random((hidden_size1, hidden_size2))
        self.w3 = np.random.random((hidden_size2, output_size))

        self.u1 = np.zeros((T_max, hidden_size1))
        self.u2 = np.zeros((T_max, hidden_size2))
        self.u3 = np.zeros((T_max, output_size))

        self.prev_t_0 = np.zeros(input_layer_size)
        self.prev_t_1 = np.zeros(hidden_size1)
        self.prev_t_2 = np.zeros(hidden_size2)
        self.prev_t_3 = np.zeros(output_size)

        self.spike_matrix1 = np.zeros((T_max, hidden_size1))
        self.spike_matrix2 = np.zeros((T_max, hidden_size2))
        self.spike_matrix3 = np.zeros((T_max, output_size))

        self.u_spike = u_spike
        self.T_max = T_max

    def apply(self, input_layer, t):
        def apply_inner_layer(input_size, output_size, prev_t_n, prev_t_n_1, u, w, d, spike_matrix):
            for i in range(output_size):
                u[t, i] = eta(t - prev_t_n_1[i])
                for j in range(input_size):
                    u[t, i] += w[j, i] * eps(t - prev_t_n[j] - d[j, i], d[j, i])
                if u[t, i] > self.u_spike:
                    prev_t_n_1[i] = t
                    spike_matrix[t][i] = 1

        for i in range(self.input_size):
            if input_layer[i] == 1:
                self.prev_t_0[i] = t

        apply_inner_layer(self.input_size, self.hidden_size1,
                          self.prev_t_0, self.prev_t_1,
                          self.u1, self.w1, self.d1, self.spike_matrix1)

        apply_inner_layer(self.hidden_size1, self.hidden_size2,
                          self.prev_t_1, self.prev_t_2,
                          self.u2, self.w2, self.d2, self.spike_matrix2)

        apply_inner_layer(self.hidden_size2, self.output_size,
                          self.prev_t_2, self.prev_t_3,
                          self.u3, self.w3, self.d3, self.spike_matrix3)

    def visualize_most_spiking_u(self):
        u_matrix_tuple = (
            self.u1,
            self.u2,
            self.u3
        )
        layer_names_tuple = (
            'first hidden layer',
            'second hidden layer',
            'output layer'
        )

        for layer_index, spike_matrix in enumerate((self.spike_matrix1, self.spike_matrix2, self.spike_matrix3)):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 10))
            spikes_count = np.sum(spike_matrix, axis=1)
            
            most_spiking_neuron_i = np.argmax(np.sum(spike_matrix, axis=0))
            u = u_matrix_tuple[layer_index]

            tim = np.arange(self.T_max)
            make_plot(ax1, tim, u[:self.T_max, most_spiking_neuron_i], color='g',
                      label='u(t) for most spiking neuron with index {} from {})'.format(most_spiking_neuron_i,
                                                                                       layer_names_tuple[layer_index]),
                      xlabel='time', ylabel='u')
            make_plot(ax2, tim, spikes_count[:self.T_max], color='b',
                      label='number of spikes in time for {}'.format(layer_names_tuple[layer_index]),
                      xlabel='time', ylabel='num spikes')

            plt.show()
            plt.close()
