from src.linear_models import LinePredictorCSI
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def mse(y, y_hat):
    return np.mean(np.power(y - y_hat, 2))


CSI_MATRIX_FILE_NAME = '..\\csi_matrix\\data\\chan_matrix_converted_to_complex.file'

with open(CSI_MATRIX_FILE_NAME, 'rb') as file:
    data = pickle.load(file)

data.real *= 10**6
data.imag *= 10**6


plt.plot(data[0, 0, 0, :, 1], label='t2')
plt.plot(data[0, 1, 0, :, 1], label='t1')
plt.legend()
plt.show()

lp = LinePredictorCSI()
lp.fit(data)

predict = lp.predict(start=3, horizon=data.shape[1] - 3)

print(mse(data[:, 3:, :, :, :].real, predict.real))
print(mse(data[:, 3:, :, :, :].imag, predict.imag))

plt.plot(predict[0, :, 0, 0, 0].real, label='predict')
plt.plot(data[0, 3:, 0, 0, 0].real, label='orig')
plt.legend()
plt.show()


sns.displot(data.ravel().imag)
plt.show()