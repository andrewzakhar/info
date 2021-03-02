import csv
import itertools
import time
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from src import MinkowskiModel

warnings.filterwarnings('ignore')
N = 3125
np.random.seed(11)


def gen_x_v1():
    X = np.zeros((N, 5))
    psi = np.random.uniform(0, np.pi / 2, size=N)
    phi = np.random.uniform(0, np.pi / 2, size=N)
    hi = np.random.uniform(0, np.pi / 2, size=N)
    t = np.random.uniform(0, np.pi / 2, size=N)

    X[:, 0] = np.cos(psi) * np.cos(phi) * np.cos(hi) * np.cos(t)
    X[:, 1] = np.cos(psi) * np.sin(phi) * np.cos(hi) * np.cos(t)
    X[:, 2] = np.sin(psi) * np.cos(hi) * np.cos(t)
    X[:, 3] = np.sin(hi) * np.cos(t)
    X[:, 4] = np.sin(t)
    return X


def gen_x_v2():
    return np.random.normal(0, 1, size=(N, 5))


def main(eps_start, eps_end, eps_edge_size, distance_metric):
    model = MinkowskiModel(eps_start=eps_start, eps_end=eps_end,
                           eps_edge_size=eps_edge_size, distance_metric=distance_metric, verbose=False)
    start = time.time()
    try:
        val, _ = model.fit_predict(x)
        return val, time.time() - start
    except ValueError:
        return None, None


if __name__ == '__main__':
    x = gen_x_v2()

    edges_sizes = np.linspace(0.05, 0.5, 10)
    eps_start_grid = np.linspace(0.05, 0.5, 10)
    eps_end_grid = np.linspace(0.05, 0.5, 10)

    metrics = ['euclidean', 'minkowski', 'seuclidean', 'chebyshev']
    with open(f'results_{datetime.now():%m_%d_%H%M}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['eps_start_grid', 'eps_end_grid', 'edges_sizes', 'metrics', 'vals', 'times'])
        pbar = tqdm(list(itertools.product(eps_start_grid, eps_end_grid, edges_sizes, metrics)))
        for params in pbar:
            if params[0] > params[1]:  # skip incorrect range
                continue
            pbar.set_description(f'{params}')
            v, tm = main(*params)
            writer.writerow([p for p in params] + [v, tm])
