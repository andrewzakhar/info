import h5py
import numpy as np

# (1352, 512, 4, 4)
# (1352, 512, 8, 4)
if __name__ == '__main__':
    data = h5py.File('path/to/file', 'r')
    arr: np.ndarray = data['h']['value'][()]  # extract all data from datapoint
    print(arr.shape)
