import time

import numpy as np

from src.data_removing import ExtendedLOF
from src.graphic_search import GraphicSearch
from src import MinkowskiModel

np.random.seed(11)
psi = np.random.uniform(0, np.pi/2, 500)
phi = np.random.uniform(0, 2 * np.pi, 500)
x1 = np.cos(psi) * np.cos(phi)
x2 = np.cos(psi) * np.sin(phi)
x3 = np.sin(psi)
x = np.array((x1, x2, x3)).T

model = GraphicSearch(reduce_input=True, verbose=True)
start = time.time()
print(model.fit_predict(x))
print(model.ranges)
print(f'elapsed time {time.time() - start}')
