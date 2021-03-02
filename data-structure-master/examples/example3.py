"""
    Example for Minkowski model with a LOF for reducing dataset
"""

from src import MinkowskiModel
from src.data_removing import LOF, ExtendedLOF

import numpy as np

np.random.seed(11)
N = 6000
x = np.zeros((N, 5))
psi = np.random.uniform(0, np.pi/2, size=N)
phi = np.random.uniform(0, np.pi/2, size=N)
hi = np.random.uniform(0, np.pi/2, size=N)
t = np.random.uniform(0, np.pi/2, size=N)

x[:, 0] = np.cos(psi)*np.cos(phi)*np.cos(hi)*np.cos(t)
x[:, 1] = np.cos(psi)*np.sin(phi)*np.cos(hi)*np.cos(t)
x[:, 2] = np.sin(psi)*np.cos(hi)*np.cos(t)
x[:, 3] = np.sin(hi)*np.cos(t)
x[:, 4] = np.sin(t)

model = MinkowskiModel(remover=ExtendedLOF())
print(f'classes = {model.fit_predict(x)}')
