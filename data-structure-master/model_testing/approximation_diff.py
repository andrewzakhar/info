import time

import numpy as np

from src import MinkowskiModel
from src.graphic_search import GraphicSearch

np.random.seed(11)
N = 500

psi = np.random.uniform(0, np.pi/2, N)
phi = np.random.uniform(0, 2 * np.pi, N)
x = np.array((np.cos(psi) * np.cos(phi), np.cos(psi) * np.sin(phi), np.sin(psi))).T

x2 = np.zeros((N, 5))
psi = np.random.uniform(0, np.pi/2, size=N)
phi = np.random.uniform(0, np.pi/2, size=N)
hi = np.random.uniform(0, np.pi/2, size=N)
t = np.random.uniform(0, np.pi/2, size=N)

x2[:, 0] = np.cos(psi)*np.cos(phi)*np.cos(hi)*np.cos(t)
x2[:, 1] = np.cos(psi)*np.sin(phi)*np.cos(hi)*np.cos(t)
x2[:, 2] = np.sin(psi)*np.cos(hi)*np.cos(t)
x2[:, 3] = np.sin(hi)*np.cos(t)
x2[:, 4] = np.sin(t)


for ds in [x, x2]:
    m = MinkowskiModel()
    start = time.time()
    predict = m.primary_dimensions(ds)
    print(f'train data = {predict}, target = {[i for i in range(ds.shape[1]) if i not in predict]} time: {time.time() - start}')

    # is graphic model
    m2 = GraphicSearch()
    start = time.time()
    target = m2.fit_predict(ds)
    print(f'train data = {[i for i in range(ds.shape[1]) if i != target]}, target = {target} time: {time.time() - start}')

