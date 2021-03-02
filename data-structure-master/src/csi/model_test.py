import os
import time
from os.path import dirname
import sys
import matplotlib.pyplot as plt
import numpy as np

from .metrics import mse, frobenius_error
from .models import BaseModel, DummyModel


class CSIData:
    tti = 300
    ue = 30
    rb = 50
    sub_ue = 64
    ch = 2

    def __init__(self, path_to_data=None, speed=100, adjust=10**6):
        self.speed = speed
        if not path_to_data:
            self.path_to_data = os.path.join(dirname(dirname(dirname(__file__))), 'csi_matrix', 'data', f'ChannelQriga_Freq_{self.speed}kmh_Scena0_test_3GPP_3D_UMa_NLOS_2p1GHz_numpy')
        else:
            self.path_to_data = path_to_data
        with open(self.path_to_data, 'rb') as file:
            self.csi_matrix = np.fromfile(file, dtype=np.complex).reshape(self.ue, self.tti, self.rb, self.sub_ue, self.ch)
            self.csi_matrix.real *= adjust
            self.csi_matrix.imag *= adjust
        self.adjust = adjust

    def __len__(self):
        return self.csi_matrix.shape[1]

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= self.tti:
            return self.csi_matrix[:, self.n, :, :, :]
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self.csi_matrix[:, item, :, :, :]


class ChannelEFPD:
    """
    Channel encoding fit predict decoding
    """

    losses = {
        'mse': mse
    }

    def __init__(self, model=BaseModel(), update_frequency=None, encoding_frequency=None, prediction_start=None,
                 loss='mse',
                 prediction_score='mse', csi_repository=CSIData()):
        # data config
        self.csi_repository = csi_repository

        # model config
        self.model = model
        self.update_frequency = update_frequency
        self.encoding_frequency = encoding_frequency
        self.prediction_start = prediction_start
        self.loss = loss
        self.prediction_score = prediction_score

    def test(self):

        space_complexity = 0
        time_complexity = 0
        loss = 0
        previous_matrix_on_receiver = None
        cold_start_receiver_buffer = []
        a = []
        for t in range(len(self.csi_repository)):  # time loop
            print(f'time : {t}')
            if self.encoding_frequency and t % self.encoding_frequency == 0:
                print('\t Encoding')
                start_time = time.time()
                matrix = self.model.encode(self.csi_repository[t])
                time_complexity += time.time() - start_time
                space_complexity += matrix.nbytes
                if self.update_frequency and t < self.prediction_start:
                    cold_start_receiver_buffer.append(matrix)
                    # skip prediction
                elif self.update_frequency and t == self.prediction_start:
                    cold_start_receiver_buffer.append(matrix)
                    print('\t\t Fitting (cold)')
                    start_time = time.time()
                    self.model.fit(cold_start_receiver_buffer)
                    time_complexity += time.time() - start_time
                elif self.update_frequency and t % self.update_frequency == 0 and t > self.prediction_start:
                    print('\t\t Fitting (regular)')
                    start_time = time.time()
                    self.model.fit((previous_matrix_on_receiver, matrix))
                    time_complexity += time.time() - start_time
                    # save matrix for predictions
                elif self.update_frequency and t > self.prediction_start:
                    print('\t\t Predicting')
                    space_complexity -= matrix.nbytes
                    start_time = time.time()
                    matrix = self.model.predict(previous_matrix_on_receiver)
                    time_complexity += time.time() - start_time
                print('\t\t\t Decoding')
                previous_matrix_on_receiver = matrix.copy()
                start_time = time.time()
                matrix = self.model.decode(matrix)
                time_complexity += time.time() - start_time
                a.append(matrix[0, 0, 0, 0].real)
                loss += self.losses[self.loss](self.csi_repository[t], matrix)
                # account all memory and computation metrics
            else:
                matrix = self.csi_repository[t]
                space_complexity += self.csi_repository[t].nbytes
        print(f'[space_complexity={space_complexity:.2f}]\n\t[time_complexity: {time_complexity:.2f}]\n\t[loss * 10^6: {loss / len(self.csi_repository):.4f}]')
        plt.plot([self.csi_repository[i][0, 0, 0, 0].real for i in range(len(self.csi_repository))], label='val')
        plt.plot(a, label='a')
        plt.legend()
        plt.show()


def recursive_len(item):
    if type(item) == list or type(item) == tuple or type(item) == np.ndarray:
        return sum(recursive_len(sub_item) for sub_item in item)
    else:
        return 1

class ChannelED:
    """
    Channel encoding decoding
    """

    losses = {
        'mse': mse,
        'frobenius': frobenius_error
    }

    def __init__(self, model=BaseModel(), encoding_frequency=1,
                 loss='mse', csi_repository=CSIData(), verbose=False):
        #
        self.verbose = verbose

        # data config
        self.csi_repository = csi_repository

        # model config
        self.model = model
        self.encoding_frequency = encoding_frequency
        self.loss = loss

    def test(self):
        raw_space_complexity = []
        compression_ratio = []  # passed / original length
        time_complexity = []
        loss = []
        for t in range(len(self.csi_repository)):  # time loop
            time_complexity.append([])
            if self.verbose:
                print(f'time : {t}')
            if t % self.encoding_frequency == 0:
                if self.verbose:
                    print('\t Encoding')
                start_time = time.time()
                matrix = self.model.encode(self.csi_repository[t])
                time_complexity[-1].append(time.time() - start_time)
                raw_space_complexity.append(sys.getsizeof(matrix))
                compression_ratio.append(recursive_len(matrix) / np.prod(self.csi_repository[t].shape))
                if self.verbose:
                    print('\t\t Decoding')
                start_time = time.time()
                matrix = self.model.decode(matrix)
                time_complexity[-1].append(time.time() - start_time)
                loss.append(self.losses.get(self.loss, 'mse')(self.csi_repository[t], matrix))
                # account all memory and computation metrics
        if self.verbose:
            print(f'loss * 10^6 = {np.sum(loss)}; avg loss * 10^6 = {np.mean(loss)}')
            print(f'time = {np.sum(time_complexity)}; avg time = {np.mean(time_complexity)}')
            print(f'compression ratio [(passed / origin) / iter] = {np.mean(compression_ratio)}')
            print(f'raw space complexity [byte/iter] = {np.mean(raw_space_complexity)}')
        return {'loss': loss, 'time': time_complexity, 'compression_ratio': compression_ratio,
                'rsc': raw_space_complexity}

if __name__ == '__main__':
    channel = ChannelEFPD(model=DummyModel(), update_frequency=10, encoding_frequency=1, prediction_start=5)
    channel.test()

    channel2 = ChannelED(model=DummyModel(), encoding_frequency=1)
    channel2.test()
