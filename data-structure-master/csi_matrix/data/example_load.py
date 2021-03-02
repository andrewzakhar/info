import numpy as np

with open('ChannelQriga_Freq_100kmh_Scena0_test_3GPP_3D_UMa_NLOS_2p1GHz_numpy', 'rb') as file:
    csi_matrix: np.ndarray = np.fromfile(file, dtype=np.complex).reshape(30, 300, 50, 64, 2)

print(f'Shape: {csi_matrix.shape}\nData Type: {csi_matrix.dtype}')
print(f'Number of Ue: {csi_matrix.shape[0]}')
print(f'Number of snapShots (TTI): {csi_matrix.shape[1]}')
print(f'Number of Rb (Resource Block): {csi_matrix.shape[2]}')
print(f'Number of BS: {csi_matrix.shape[3]}')
print(f'Number of ?Channels? {csi_matrix.shape[4]}')


print(csi_matrix.max())
print(csi_matrix.min())