import numpy as np


def mse(y_true, y_predicted):
    return np.mean(np.power(y_true.real - y_predicted.real, 2)) + np.mean(np.power(y_true.imag - y_predicted.imag, 2))


def frobenius_norm(matrix):
    return np.sqrt(np.sum(np.power(np.abs(matrix), 2)))


def frobenius_error(y_true, y_predicted):
    return frobenius_norm(y_true - y_predicted)
