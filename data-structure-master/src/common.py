import itertools
import time

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset


def __subcubes_number(grid, x, eps):
    result = 0
    for item in grid:
        if ((item <= x) & (x < item + eps)).all(axis=1).any():
            result += 1
    return result


def __batch_on_iter(iterable: iter, length: int, size: int = 1):
    for index in range(0, length, size):
        yield np.array([next(iterable) for _ in range(index, min(index + size, length))])


def _find_number_of_spheres(x: np.ndarray, eps: float, n_jobs=-1, verbose=True) -> int:
    """
    TODO: add time estimate
    :param x: np.array[np.float64]
    :param eps: float
    :return: minimum number of spheres which covers all points
    """
    dimensions = x.shape[1]
    points = [np.arange(0, 1, eps) for _ in range(dimensions)]
    if verbose:
        print(f'creating grid {points[0].shape} len={points[0].shape[0] ** dimensions}')
    grid = itertools.product(*points)
    grid_length = points[0].shape[0] ** dimensions
    parallel = Parallel(n_jobs=n_jobs, prefer='processes')
    number_of_chunks = parallel._effective_n_jobs()
    batch_size = 67108864  # math.ceil(grid_length / 10)  # TODO: estimate the memory
    result = 0
    start_time = time.time()
    for batch in __batch_on_iter(grid, grid_length, batch_size):
        result += sum(parallel(delayed(__subcubes_number)(p_batch, x, eps) for p_batch in np.array_split(batch, number_of_chunks)))
    if verbose:
        print(f'elapsed time {time.time() - start_time}')
    return result


def _distances(x: np.ndarray, distance_metric='euclidean') -> np.ndarray:
    distances = cdist(x, x, metric=distance_metric)
    np.fill_diagonal(distances, np.nan)
    return distances


def _euc_distance(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.sum(np.power(x[np.newaxis, :] - y, 2), axis=1))


def is_graphic(x: np.ndarray, dimension: int, distance_metric='euclidean', left_bound=-1., right_bound=1.,
               divider=2):
    distances = _distances(x, distance_metric=distance_metric)
    min_distance = np.nanmin(distances)
    counter = 0
    for i in range(x.shape[0]):
        # if memory is not critical we can create a grid for p and b and increase speed
        # estimated_finish = -np.sqrt(min_distance ** 2 - _euc_distance(x[i][x_indices], x[i][x_indices])) + right_bound
        p, b = x[i].copy(), x[i].copy()
        p[dimension] = left_bound
        b[dimension] = right_bound
        while _euc_distance(p, b) >= min_distance:
            e = _euc_distance(p, np.delete(x, i, axis=0))
            k = np.where(e < min_distance)
            counter += k[0].shape[0]
            p[dimension] += min_distance / divider
    return counter


class LinearDataset(Dataset):
    def __init__(self, x: np.ndarray, device, target_idx=None):
        self.x = x
        self.target_idx = target_idx
        self.train_columns = list(range(x.shape[1]))
        if target_idx:
            if isinstance(target_idx, list):
                for item in target_idx:
                    self.train_columns.remove(item)
            else:
                self.train_columns.remove(target_idx)
        self.device = device

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.target_idx:
            return torch.tensor(self.x[idx, self.train_columns], device=self.device, dtype=torch.float),\
                   torch.tensor(self.x[idx, self.target_idx], device=self.device, dtype=torch.float)
        else:
            return torch.tensor(self.x[idx, self.train_columns], device=self.device, dtype=torch.float)
