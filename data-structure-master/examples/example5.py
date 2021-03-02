from src.common import is_graphic, is_graphic2
import numpy as np
import time


np.random.seed(11)
psi = np.random.uniform(0, np.pi/2, 500)
phi = np.random.uniform(0, 2 * np.pi, 500)
x1 = np.cos(psi) * np.cos(phi)
x2 = np.cos(psi) * np.sin(phi)
x3 = np.sin(psi)
x = np.array((x1, x2, x3)).T

results = np.zeros(10)
for i in range(10):
    print(f'step i={i}')
    start = time.time()
    print(is_graphic(x, 2, divider=2))
    results[i] = time.time() - start

print(f'mean = {np.mean(results)}')
