import math
import warnings
from collections import namedtuple

import numpy as np
from joblib import Parallel, delayed

from ..clustering import Polynomials

warnings.filterwarnings('ignore')

Line = namedtuple('Line', ['k', 'b'])


class BaseModel:
    def __init__(self):
        pass

    def fit(self, x: np.ndarray):
        """

        :param x: numpy complex matrix with shape (30, 50, 64, 2)
        :return: None -> you don't need to return anything
        """
        raise NotImplementedError

    def update(self, x: np.ndarray):
        """

        :param x: numpy complex matrix with shape (30, 50, 64, 2)
        :return: None -> you don't need to return anything
        """
        raise NotImplementedError

    def encode(self, x: np.ndarray):
        """

        :param x: numpy complex matrix with shape (30, 50, 64, 2)
        :return: numpy matrix in any form you want
        """
        raise NotImplementedError

    def decode(self, x: np.ndarray):
        """

        :param x: numpy matrix with shape from encode function
        :return: numpy complex matrix with shape (30, 50, 64, 2)
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray):
        """

        :param x: TODO: depends on scheme f -> e -> d -> p or e -> f -> p -> d
        :return:
        """
        raise NotImplementedError


class DummyModel(BaseModel):
    def __init__(self):
        super(BaseModel, self).__init__()

    def encode(self, X):
        return X

    def decode(self, X):
        return X

    def fit(self, X):
        pass

    def update(self, X):
        pass

    def predict(self, X):
        return X


class LinearModel(BaseModel):
    lines_array_real_ = None
    lines_array_imag_ = None

    state = 0

    def __init__(self):
        super(BaseModel, self).__init__()

    def fit(self, stacked_X):
        """

        :param stacked_X: two csi matrix
        :return:
        """

        def _multi_line_fit(y1, y2):
            delta = -np.ones_like(y2)
            k = (y1 - y2) / delta
            return Line(k, y2 - k)

        self.lines_array_real_ = _multi_line_fit(stacked_X[0].real, stacked_X[1].real)
        self.lines_array_imag_ = _multi_line_fit(stacked_X[0].imag, stacked_X[1].imag)
        self.state = 1

    def update(self, stacked_X):
        pass

    def predict(self, X):
        self.state += 1
        base = np.full((30, 50, 64, 2), self.state)
        result_real = (self.lines_array_real_.k * base) + self.lines_array_real_.b
        result_imag = (self.lines_array_imag_.k * base) + self.lines_array_imag_.b
        return result_real + 1j * result_imag

    def encode(self, X):
        return X

    def decode(self, X):
        return X


class PolyReduction(BaseModel):
    def __init__(self, deg):
        super(BaseModel, self).__init__()
        self.deg = deg
        self.polynomial_cluster = Polynomials(deg=self.deg)

    def encode(self, X):
        # labels = self.polynomial_cluster.fit(X)
        matrix = []
        labels = [1.] * 30
        for i in range(len(labels)):  # iter over ue
            if labels[i] == 1.:  # we can compress it
                sub_matrix = []
                x_ = np.arange(X.shape[1])
                for j in range(X.shape[2]):
                    chanel_matrix = []
                    for k in range(X.shape[3]):
                        y_real = np.polyfit(x_, X[i, :, j, k].real, deg=self.deg)
                        y_imag = np.polyfit(x_, X[i, :, j, k].imag, deg=self.deg)
                        chanel_matrix.append((y_real, y_imag))
                    sub_matrix.append(chanel_matrix)
                matrix.append(sub_matrix)
            else:
                matrix.append(X[i])
        return matrix

    def decode(self, X):
        matrix = []
        for i in range(len(X)):
            if isinstance(X[i], list):
                """sub_matrix = [
                    [
                        [len(2)]
                        ...
                        # 64
                    ]
                    ...
                    # 50
                ]"""
                sub_matrix = [[] for _ in range(50)]
                for n in range(50):
                    for j in range(64):
                        p0_real = np.poly1d(X[i][j][0][0])
                        p0_imag = np.poly1d(X[i][j][0][1])
                        p1_real = np.poly1d(X[i][j][1][0])
                        p1_imag = np.poly1d(X[i][j][1][1])
                        line = (p0_real(n) + 1j * p0_imag(n), p1_real(n) + 1j * p1_imag(n))
                        sub_matrix[n].append(line)
                sub_matrix = np.array(sub_matrix)
                matrix.append(sub_matrix)
            else:
                matrix.append(X[i])
        return np.array(matrix)


class SVDModel(BaseModel):

    def __init__(self, saving=None, parallel=False):
        """

        :param saving: str, int, float
            : str : ['svht', 'rank']
            : int : number of lines in singular decomposition to keep
            : float : percentage of lines to keep
        :param parallel:
        """
        super(BaseModel, self).__init__()
        self.saving = saving
        self.parallel = parallel

    @staticmethod
    def _svht_heuristic(beta):
        return 0.56 * np.power(beta, 3) - 0.95 * np.power(beta, 2) + 1.82 + 1.43

    @staticmethod
    def _knee_second_derivative(vector):
        """
        https://raghavan.usc.edu//papers/kneedle-simplex11.pdf
        :param vector:
        :return:
        """
        return np.argmax(np.diff(vector, 2) / np.power(1 + np.power(np.diff(vector), 2), 1.5)[1:])

    def _atomic_encode(self, x: np.ndarray):
        """

        :param x: 50x64
        :return:
        """
        u, e, v = np.linalg.svd(x, full_matrices=False)
        if isinstance(self.saving, float) and 0. < self.saving <= 1.:
            rank = math.ceil(self.saving * e.shape[0])
        elif isinstance(self.saving, int):
            rank = self.saving
        elif isinstance(self.saving, str) and self.saving == 'svht':
            rank = e[e > self._svht_heuristic(x.shape[0] / x.shape[1]) * np.median(e)].shape[0]
        elif isinstance(self.saving, str) and self.saving == 'rank':
            rank = np.linalg.matrix_rank(x)
        elif isinstance(self.saving, str) and self.saving == 'sd':
            rank = self._knee_second_derivative(e)
        else:
            rank = math.ceil(e.shape[0] / 2)
        u = u[:, :rank]
        v = v[:rank]
        e = e[:rank]
        return u, e, v

    def _encode_fork(self, matrix):
        return self._atomic_encode(matrix[:, :, 0]), self._atomic_encode(matrix[:, :, 1])

    def _atomic_decode(self, u, e, v):
        e = np.diag(e)
        return u.dot(e).dot(v)

    def _decode_fork(self, matrix):
        return np.array([self._atomic_decode(*matrix[0]), self._atomic_decode(*matrix[1])]).reshape((50, 64, 2))

    def encode(self, x: np.ndarray):
        if self.parallel:
            encoded_matrix = Parallel(n_jobs=-1, prefer='processes')(delayed(self._encode_fork)(x[ue]) for ue in range(x.shape[0]))
        else:
            encoded_matrix = []
            for ue in range(x.shape[0]):
                encoded_matrix.append((
                    self._atomic_encode(x[ue, :, :, 0]),
                    self._atomic_encode(x[ue, :, :, 1]),
                ))
        return encoded_matrix

    def decode(self, x):
        if self.parallel:
            matrix = Parallel(n_jobs=-1, prefer='processes')(delayed(self._decode_fork)(x[ue]) for ue in range(len(x)))
        else:
            matrix = []
            for ue in range(len(x)):
                matrix.append(np.dstack((
                    np.array(self._atomic_decode(*x[ue][0])),
                    np.array(self._atomic_decode(*x[ue][1]))
                )))
        return np.array(matrix)


if __name__ == '__main__':
    from .model_test import CSIData
    from .metrics import frobenius_error

    rep = CSIData(speed=40)
    m = PolyReduction(deg=8)
    loss = 0
    for i in range(len(rep)):
        print(f'process {i}')
        res = m.encode(rep[i])
        res2 = m.decode(res)
        loss += frobenius_error(rep[i], res2)
    print(loss)
