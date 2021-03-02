import time

import numpy as np

from src.graphic_search import GraphicSearchPipeLine

np.random.seed(11)
psi = np.random.uniform(0, np.pi/2, 500)
phi = np.random.uniform(0, 2 * np.pi, 500)
x1 = np.cos(psi) * np.cos(phi)
x2 = np.cos(psi) * np.sin(phi)
x3 = np.sin(psi)
x = np.array((x1, x2, x3)).T

model = GraphicSearchPipeLine(n_epoch=250)
start = time.time()
model.run(x)
print(f'elapsed time {time.time() - start}')