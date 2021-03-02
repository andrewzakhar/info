import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.clustering import Polynomials
sns.set()

if __name__ == '__main__':
    with open('..\\..\\csi_matrix\\data\\ChannelQriga_Freq_3kmh_Scena0_test_3GPP_3D_UMa_NLOS_2p1GHz_numpy', 'rb') as file:
        csi_matrix = np.fromfile(file, dtype=np.complex).reshape(30, 300, 50, 64, 2)

    csi_matrix.real *= 10 ** 6
    csi_matrix.imag *= 10 ** 6

    model = Polynomials(deg=8)
    print(model.fit(csi_matrix[:, 0, :, :, :]))
    print(len([i for i in model.labels_ if i == 1.]) / len([i for i in model.labels_ if i == 0.]))
    # plt.plot(model.labels_)
    # plt.show()


