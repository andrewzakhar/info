import warnings

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .common import LinearDataset
from .common import is_graphic
from .data_removing import ExtendedLOF
from .dimension_search import MinkowskiModel
from .networks import BaseNet


class GraphicSearch:
    def __init__(self, distance_metric='euclidean', increment_coef=2, max_depth=10, left_bound=-1,
                 right_bound=1, reduce_input=False, verbose=False):
        self.distance_metric = distance_metric
        self.increment_coef = increment_coef
        self.max_depth = max_depth
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.reduce_input = reduce_input

        self.verbose = verbose

        self._depth = None

    @staticmethod
    def _gen_slices(left_bound, right_bound, depth):
        if depth == 1:
            return [[left_bound, right_bound]]
        delta = (right_bound - left_bound) / depth
        result = [[left_bound, left_bound + delta]]
        for i in range(1, depth - 1):
            result.append([result[-1][1], result[-1][1] + delta])
        result.append([result[-1][1], right_bound])
        return result

    def _grid_depth(self, x, column_idx, left_bound, right_bound):
        for i in range(2, self.max_depth, 2):
            if self.verbose:
                print(f'depth search step = {i}')
            counters = [is_graphic(x, column_idx, left_bound=l_bound, right_bound=r_bound, divider=self.increment_coef,
                                   distance_metric=self.distance_metric)
                        for l_bound, r_bound in
                        self._gen_slices(left_bound=left_bound, right_bound=right_bound, depth=i)]
            if self.verbose:
                print(f'count on each sub sets = {counters}')
            if sum(counters) == 0:
                return i
        warnings.warn(f'the max_grid is reached. Using the max value')
        return self.max_depth

    def fit_predict(self, x: np.ndarray):
        if self.reduce_input:
            x = ExtendedLOF(reduce_coef=0.2).fit_delete(x)
        n_min = x.shape[0]
        target_column = 0
        for column_idx in range(x.shape[1]):
            if self.verbose:
                print(f'checking {column_idx} column')
            n = is_graphic(x, column_idx, distance_metric=self.distance_metric,
                           divider=self.increment_coef, left_bound=self.left_bound,
                           right_bound=self.right_bound)
            if n < n_min:
                n_min = n
                target_column = column_idx
            if n_min == 0:
                self._depth = 1
                return target_column
        self._depth = self._grid_depth(x, target_column, self.left_bound, self.right_bound)
        return target_column

    @property
    def ranges(self):
        return self._gen_slices(self.left_bound, self.right_bound, self._depth) if self._depth else None


class GraphicSearchPipeLine:
    def __init__(self, dimension_search_model=None, graphic_search_model=None, neural_network_model=None,
                 batch_size=50, n_epoch=20, test_train_proportion=0.8, verbose=False):
        self.dimension_search_model = dimension_search_model if dimension_search_model else MinkowskiModel()
        self.graphic_search_model = graphic_search_model if graphic_search_model else GraphicSearch(reduce_input=True)
        self.neural_network_model = neural_network_model

        self.batch_size = batch_size
        self.n_epoch = n_epoch
        if test_train_proportion < .1 or test_train_proportion > .9:
            raise ValueError('the values should be in range [0.1, 0.9] for test_train_proportion')
        else:
            self.test_train_proportion = test_train_proportion

        self.verbose = verbose

    def run(self, x):
        mean_dimension, predict = self.dimension_search_model.fit_predict(x)
        if x.shape[1] - 1 != predict:
            raise Exception(f'Not Implement behavior predicted shape = {predict} expected = {x.shape[1] - 1}')

        target_column = 2  # self.graphic_search_model.fit_predict(x)
        if not self.neural_network_model:
            self.neural_network_model = BaseNet(dimension_size=x.shape[1] - 1)

        # train network
        optimizer = optim.Adam(self.neural_network_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        test_size = int(x.shape[0] * self.test_train_proportion)  # get holdout part
        train = LinearDataset(x=x[:-test_size], ranges=self.graphic_search_model.ranges, target_idx=target_column,
                              device=self.neural_network_model.device)
        test = LinearDataset(x=x[-test_size:], ranges=self.graphic_search_model.ranges, target_idx=target_column,
                             device=self.neural_network_model.device)
        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        for i in range(self.n_epoch):
            if self.verbose:
                print(f'training network. epoch number = {i}')
            for batch_id, batch in enumerate(train_loader):
                self.neural_network_model.train()
                optimizer.zero_grad()
                out = self.neural_network_model(batch[0])
                loss = criterion(out, batch[1].view(out.shape[0], 1))
                loss.backward()
                optimizer.step()

        # evaluating the model
        test_loss = []
        for j in range(len(test)):
            _x, _y = test[j][0].unsqueeze(0), test[j][1]
            self.neural_network_model.eval()
            response = self.neural_network_model(_x)
            loss = criterion(response, _y.view(response.shape[0], 1))
            test_loss.append(loss.detach().numpy())
        print(np.mean(test_loss))

        return True
