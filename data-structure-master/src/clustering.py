import numpy as np


class Polynomials:
    labels_ = None

    def __init__(self, alpha=1., deg=8):
        self.alpha = alpha
        self.deg = deg + 1

    def fit(self, X):
        """

        :param X: csi matrix for time t :: shape is (Ue, RB, subUe, ch) [real/imag]
        :return:
        """
        self.labels_ = np.zeros(len(X))
        for i in range(len(X)):
            std = np.diff(X[i, :, :, :].real, axis=0, n=self.deg).std() + np.diff(X[i, :, :, :].imag, axis=0, n=self.deg).std()
            if std < self.alpha:
                self.labels_[i] = 1.
        return self.labels_


if __name__ == '__main__':
    with open('..\csi_matrix\data\ChannelQriga_Freq_100kmh_Scena0_test_3GPP_3D_UMa_NLOS_2p1GHz_numpy', 'rb') as file:
        csi_matrix: np.ndarray = np.fromfile(file, dtype=np.complex).reshape(30, 300, 50, 64, 2)

    model = Polynomials(deg=8)
    print(model.fit(csi_matrix[:, 0, :, :, :]))
