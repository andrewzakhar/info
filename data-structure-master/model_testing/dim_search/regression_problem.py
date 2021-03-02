from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from src import MinkowskiModel, FractalModel
import time
import numpy as np

np.random.seed(11)


data = make_regression(n_samples=10000, n_features=10, n_informative=3, effective_rank=4000, n_targets=1, random_state=11)

X = data[0]

scaler = MinMaxScaler()
scaler.fit(X)
x = scaler.transform(X)

model = FractalModel(distance_metric='chebyshev', f_steps=20, verbose=True)
start = time.time()
result = model.fit_predict(x)
end = time.time() - start
print(f'elapsed time {end} result: {result}')
