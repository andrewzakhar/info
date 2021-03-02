from collections import namedtuple

import numpy as np

Line = namedtuple('Line', ['k', 'b'])


def mse(y, y_hat):
    return np.sum(np.power(y.real - y_hat.real, 2)) + np.sum(np.power(y.imag - y_hat.imag), 2)


def mpe(y, y_hat):
    scale = 100 / np.product(y.shape)
    return abs(scale * np.sum((y.real - y_hat.real) / y.real)) + abs(scale * np.sum((y.imag - y_hat.imag) / y.imag))


def get_loss(csi_slice, t, real_coef: Line, imag_coef: Line, king='mse'):
    x = np.full((30, 50, 64, 2), t)
    real_predict = real_coef.k * x + real_coef.b
    imag_predict = imag_coef.k * x + imag_coef.b
    if king == 'mse':
        return mse(csi_slice, real_predict + 1j * imag_predict)
    elif king == 'mpe':
        return mpe(csi_slice, real_predict + 1j * imag_predict)


class LinePredictorCSI:
    def __init__(self):
        self.fitted_lines_real = None
        self.fitted_lines_imag = None

    def _fit_line_m(self, y1, y2, start=None):
        """

        Args:
            y1:
            y2:
            start:

        Returns:

        """
        if not start:
            if y1.shape != y2.shape:
                raise Exception(f'The shapes are not matching {y1.shape}!={y2.shape}')
            delta = -np.ones_like(y2)  # equivalent for formula (y1 - y2) / (x1 - x2)
            k = (y1 - y2) / delta
            return Line(k, y2 - k)
        else:
            return Line(0, 0)

    def fit(self, x):
        """

        Args:
            x: shape 30 x 2 x 50 x 64 x 2

        Returns: None

        """
        self.fitted_lines_real = self._fit_line_m(x[:, 0, :, :, :].real, x[:, 1, :, :, :].real)
        self.fitted_lines_imag = self._fit_line_m(x[:, 0, :, :, :].imag, x[:, 1, :, :, :].imag)

    def predict(self, start=3, horizon=100):
        base = np.full((30, 50, 64, 2), start)
        result_real = np.zeros((30, horizon, 50, 64, 2))
        result_imag = np.zeros((30, horizon, 50, 64, 2))
        for i in range(horizon):
            result_real[:, i, :, :, :] = self.fitted_lines_real.k * (base + i) + self.fitted_lines_real.b
            result_imag[:, i, :, :, :] = self.fitted_lines_imag.k * (base + i) + self.fitted_lines_imag.b
        return result_real + 1j * result_imag
