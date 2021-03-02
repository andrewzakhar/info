import itertools
import math as m
import warnings

import numpy as np

from .common import _distances, _find_number_of_spheres
from .data_removing import DBF


class MinkowskiModel:
    """

    """

    def __init__(self, eps_start=0.2, eps_end=None, eps_tol=0.05, h=0.05, eps_edge_size=0.005,
                 distance_metric='euclidean', sample_min_size=None,
                 f_steps=5, f_chunks_number=12, f_increase_coef=3, f_point_shift=0.5,
                 r_steps=5, remover=DBF(reduce_coef=0.1),
                 verbose=False, store_dataset=True, n_jobs=-2):
        self.h = h
        self.eps_start = eps_start
        self.eps_end = eps_end if eps_end else 1 / 3 - self.h  # ~0.28
        self.eps_edge_size = eps_edge_size
        self.eps_tol = eps_tol

        self.distance_metric = distance_metric
        self.sample_min_size = sample_min_size if sample_min_size else 1.5 * pow(5, 5)

        self.f_steps = f_steps
        self.f_chunks_number = f_chunks_number
        self.f_increase_coef = f_increase_coef
        self.f_point_shift = f_point_shift

        self.r_steps = r_steps
        self.remover = remover

        self.verbose = verbose
        self.store_dataset = store_dataset

        self.n_jobs = n_jobs

        self._x_removed = None
        self._x_filled = None

    def _desired_dimension(self, x: np.ndarray, eps: float, h: float) -> float:
        return (m.log(_find_number_of_spheres(x, eps + h, n_jobs=self.n_jobs, verbose=self.verbose))
                - m.log(_find_number_of_spheres(x, eps, n_jobs=self.n_jobs, verbose=self.verbose))) \
               / (m.log(eps) - m.log(eps + h))

    def gen_points(self, x: np.ndarray, distances: np.ndarray, nearest_distances: np.ndarray) -> np.ndarray:
        k0 = x.shape[0] // self.f_chunks_number
        k_max_nearest = np.sort(nearest_distances, kind='mergesort')[-k0:]
        indices = np.where(np.isin(distances, k_max_nearest))  # fix indices[0]
        additional_data = np.zeros((k0, self.f_increase_coef, x.shape[1]), dtype=np.float64)
        for i in range(k0):
            j = distances[indices[1][i]].argsort()[:self.f_increase_coef]
            additional_data[i] = self.f_point_shift * x[[indices[0][i] for _ in range(self.f_increase_coef)]] \
                                 + (1 - self.f_point_shift) * x[j]
        return additional_data.reshape(k0 * self.f_increase_coef, x.shape[1])

    def fit_predict(self, x: np.ndarray) -> (float, int):
        x.astype(dtype=np.float64, copy=False)
        distances = _distances(x, self.distance_metric)
        nearest_distances = np.nanmin(distances, axis=1)
        max_nearest_distance = np.nanmax(nearest_distances)
        if self.verbose:
            print(f'Mean distance {np.mean(nearest_distances)} Max distance: {max_nearest_distance}')

        # removing part
        step = 0
        while 4 * np.mean(nearest_distances) <= max_nearest_distance and step < self.r_steps:
            if self.verbose:
                print(
                    f'removing >> \n\tmean distance {np.mean(nearest_distances)} \tmax distance: {max_nearest_distance}')
            x = self.remover.fit_delete(x, nearest_distances)
            distances = _distances(x, self.distance_metric)
            nearest_distances = np.nanmin(distances, axis=1)
            max_nearest_distance = np.nanmax(nearest_distances)
            step += 1
        if step >= self.r_steps and self.verbose:
            print('maximum steps performed')
        if self.store_dataset:
            self._x_removed = x.copy()
        # filling part
        eps_left = max(self.eps_start, max_nearest_distance)
        eps_right = self.eps_end
        step = 0
        while (eps_right - eps_left <= self.eps_tol or x.shape[0] < self.sample_min_size) and step < self.f_steps:
            if self.verbose:
                print(
                    f'filling >> \n\tmean distance {np.mean(nearest_distances)} \tmax distance: {max_nearest_distance} delta[{eps_right - eps_left}] size[{x.shape}/{self.sample_min_size}]')
            x = np.concatenate((x, self.gen_points(x, distances, nearest_distances)), axis=0)
            distances = _distances(x, self.distance_metric)
            nearest_distances = np.nanmin(distances, axis=1)
            max_nearest_distance = np.nanmax(nearest_distances)

            eps_left = max(self.eps_start, max_nearest_distance)
            step += 1
        if step >= self.f_steps and self.verbose:
            print('maximum steps performed')
        if self.store_dataset:
            self._x_filled = x.copy()
        # compute number of dimensions ceil((stop - start)/step)
        eps = np.arange(eps_left, eps_right, self.eps_edge_size)
        if self.verbose:
            print(f'search in eps[{eps}] with step {self.h}')
        predicted_dimensions = []
        for e in eps:
            if self.verbose:
                print(f'working on eps[{e}] eps with h[{self.h}]')
            predicted_dimensions.append(self._desired_dimension(x, e, self.h))
        mean_dimension = np.mean(predicted_dimensions)
        predicted_dimension = m.ceil(np.mean(mean_dimension) + 0.05)
        if self.verbose:
            print(f'\tmean dimension = {mean_dimension}')
            print(f'\tpredicted dimension = {predicted_dimension}')

        return mean_dimension, predicted_dimension

    def primary_dimensions(self, x: np.ndarray) -> list:
        _, number_of_dimensions = self.fit_predict(x)
        results = []
        if number_of_dimensions < x.shape[1]:
            for combination in itertools.combinations(range(x.shape[1]), number_of_dimensions):
                try:
                    plain_dimension, _ = self.fit_predict(x[:, combination])
                    results.append((combination, plain_dimension))
                except Exception as e:
                    if self.verbose:
                        print(e)
            if self.verbose:
                print(results)
            best_combination = [i for i in sorted(results, key=lambda e: e[1], reverse=True)[0][0]]
            return best_combination
        else:
            warnings.warn(f'the predicted sub dimension is equal to original dimension')
            return [i for i in range(x.shape[1])]


class FractalModel:
    def __init__(self, distance_metric='euclidean', sample_min_size=None,
                 f_steps=10, f_chunks_number=12, f_increase_coef=3, f_point_shift=0.5,
                 r_steps=3, remover=DBF(reduce_coef=0.1),
                 verbose=True, n_jobs=-2):

        self.distance_metric = distance_metric
        self.sample_min_size = sample_min_size if sample_min_size else 1.5 * pow(5, 5)

        self.f_steps = f_steps
        self.f_chunks_number = f_chunks_number
        self.f_increase_coef = f_increase_coef
        self.f_point_shift = f_point_shift

        self.r_steps = r_steps
        self.remover = remover

        self.n_jobs = n_jobs

        self.verbose = verbose

    def _desired_dimension(self, x: np.ndarray, eps: float) -> float:
        return m.log(_find_number_of_spheres(x, eps, n_jobs=self.n_jobs, verbose=self.verbose)) / m.log(1 / eps)

    def gen_points(self, x: np.ndarray, distances: np.ndarray, nearest_distances: np.ndarray) -> np.ndarray:
        k0 = x.shape[0] // self.f_chunks_number
        k_max_nearest = np.sort(nearest_distances)[-k0:]
        indices = np.where(np.isin(distances, k_max_nearest))  # fix indices[0]
        additional_data = np.zeros((k0, self.f_increase_coef, x.shape[1]), dtype=np.float64)
        for i in range(k0):
            j = distances[indices[1][i]].argsort()[:self.f_increase_coef]
            additional_data[i] = self.f_point_shift * x[[indices[0][i] for _ in range(self.f_increase_coef)]] \
                                 + (1 - self.f_point_shift) * x[j]
        return additional_data.reshape(k0 * self.f_increase_coef, x.shape[1])

    def fit_predict(self, x: np.ndarray) -> (None, int):
        x.astype(dtype=np.float64, copy=False)
        distances = _distances(x, self.distance_metric)
        nearest_distances = np.nanmin(distances, axis=1)
        max_nearest_distance = np.nanmax(nearest_distances)
        if self.verbose:
            print(f'Mean distance {np.mean(nearest_distances)} Max distance: {max_nearest_distance}')

        # removing part
        step = 0
        while 4 * np.mean(nearest_distances) <= max_nearest_distance and step < self.r_steps:
            if self.verbose:
                print(
                    f'removing >> \n\tmean distance {np.mean(nearest_distances)} \tmax distance: {max_nearest_distance}')
            x = self.remover.fit_delete(x, nearest_distances)
            distances = _distances(x, self.distance_metric)
            nearest_distances = np.nanmin(distances, axis=1)
            max_nearest_distance = np.nanmax(nearest_distances)
            step += 1
        if step >= self.r_steps and self.verbose:
            print('maximum steps performed')
        # filing part
        step = 0
        while x.shape[0] < self.sample_min_size and step < self.f_steps:
            if self.verbose:
                print(
                    f'filling >> \n\tmean distance {np.mean(nearest_distances)} \tmax distance: {max_nearest_distance}')
            x = np.concatenate((x, self.gen_points(x, distances, nearest_distances)), axis=0)
            distances = _distances(x, self.distance_metric)
            nearest_distances = np.nanmin(distances, axis=1)
            max_nearest_distance = np.nanmax(nearest_distances)
            step += 1
        if step >= self.f_steps and self.verbose:
            print('maximum steps performed')

        # find dimensions
        eps = max(0.1, np.max(nearest_distances))
        return None, round(self._desired_dimension(x, eps))

    def primary_dimensions(self, x: np.ndarray) -> list:
        _, number_of_dimensions = self.fit_predict(x)
        results = []
        if number_of_dimensions < x.shape[1]:
            for combination in itertools.combinations(range(x.shape[1]), number_of_dimensions):
                try:
                    plain_dimension, _ = self.fit_predict(x[:, combination])
                    results.append((combination, plain_dimension))
                except Exception as e:
                    if self.verbose:
                        print(e)
            if self.verbose:
                print(results)
            best_combination = [i for i in sorted(results, key=lambda e: e[1], reverse=True)[0][0]]
            return best_combination
        else:
            warnings.warn(f'the predicted sub dimension is equal to original dimension')
            return [i for i in range(x.shape[1])]
