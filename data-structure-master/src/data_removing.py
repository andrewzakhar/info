import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from .common import _distances


class LOF:
    """
    TODO: can be done with matrix operations
    """
    def __init__(self, tau=3., k=20, distance_metric='euclidean', reduce_coef=0.1):
        self.tau = tau
        self.k = k
        self.distance_metric = distance_metric
        self.reduce_coef = reduce_coef

    def _reachability_distance(self, first_point, second_point, number_of_neighbor):
        """
        :param first_point: index of additional point (i)
        :param second_point: index of main point (t)
        :param number_of_neighbor: the position of neighbor
        :return:
        """
        return max(self.distances[first_point, second_point],
                   self.distances[second_point, self.neighbors[second_point, number_of_neighbor]])

    def _local_reachability_density(self, point_index):
        return 1 / (sum([self._reachability_distance(k, point_index, i)
                         for i, k in enumerate(self.neighbors[point_index])]) / self.k)

    def fit_predict(self, x: np.ndarray, nearest_distances=None) -> np.ndarray:
        self.distances = _distances(x, distance_metric=self.distance_metric)
        self.nearest_distances = nearest_distances if nearest_distances.all() else np.nanmin(self.distances, axis=1)
        self.neighbors = np.array([self.distances[:, i].argsort()[:self.k] for i in range(self.distances.shape[0])])
        self._estimates = np.zeros(x.shape[0])
        for i in range(self._estimates.shape[0]):
            self._estimates[i] = sum([self._local_reachability_density(j) / self._local_reachability_density(i)
                                      for j in self.neighbors[i]]) / self.k
        return np.array(self._estimates > self.tau, dtype=np.int)

    def fit_delete(self, x: np.ndarray, nearest_distances: np.ndarray):
        k0 = round(x.shape[0] * self.reduce_coef)
        self.fit_predict(x, nearest_distances)
        return np.delete(x, self._estimates.argsort()[-k0:], axis=0)


class DBF:
    """
    Distance based filter:
        the default approach which was proposed with Minkowski and Fractal src
    """
    def __init__(self, reduce_coef=0.1):
        self.reduce_coef = reduce_coef
        pass

    def fit_delete(self, x: np.ndarray, nearest_distances: np.ndarray) -> np.ndarray:
        k0 = round(x.shape[0] * self.reduce_coef)
        indices = np.where(np.isin(nearest_distances, np.sort(nearest_distances, kind='mergesort')[-k0:]))
        return np.delete(x, indices[0], axis=0)


class ExtendedLOF(LocalOutlierFactor):
    """
    Based on scikit learn realization
    """
    def __init__(self, reduce_coef=0.1, **kwargs):
        super(ExtendedLOF, self).__init__(**kwargs)
        self.reduce_coef = reduce_coef

    def fit_delete(self, x, nearest_distances=None):
        """

        :param x:
        :param nearest_distances: stub for API
        :return:
        """
        self.fit(x)
        k0 = round(x.shape[0] * self.reduce_coef)
        return np.delete(x, self.negative_outlier_factor_.argsort()[:k0], axis=0)
