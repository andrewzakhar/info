"""
    Minkowski Model example with optimal parameters
"""


from src import MinkowskiModel
import numpy as np
import time

N = 3125
np.random.seed(11)

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
# x[:, 5] = np.linspace(0, 1, N)


model = MinkowskiModel(eps_start=0.3, eps_edge_size=0.1, eps_end=0.4, distance_metric='chebyshev', f_steps=20)
start = time.time()
number_of_dimensions = model.fit_predict(x)
print(f'elapsed time: {time.time() - start}')
print(f'number of dimensions = {number_of_dimensions}')
# print(f'the best columns = {model.primary_dimensions(x)}')
