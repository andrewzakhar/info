from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from src import MinkowskiModel, FractalModel
import time
import numpy as np

np.random.seed(11)


data = load_iris()

X = data.data

scaler = MinMaxScaler()
scaler.fit(X)
x = scaler.transform(X)

model = FractalModel(distance_metric='chebyshev', f_steps=20, verbose=True)
start = time.time()
result = model.fit_predict(x)
end = time.time() - start
print(f'elapsed time {end} result: {result}')
