import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src import MinkowskiModel

np.random.seed(11)

data = datasets.load_iris()
x = data.data
y = data.target

x = MinMaxScaler().fit_transform(x)

# find primary dimensions
main_coordinates = MinkowskiModel(verbose=False, f_steps=10, f_increase_coef=5, f_chunks_number=20).primary_dimensions(x)
target_coordinates = [i for i in range(x.shape[1]) if i not in main_coordinates]
X, Y = x[:, main_coordinates], x[:, [2, 3]]

X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(X, Y, np.arange(X.shape[0]), test_size=0.33, random_state=11)


model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

result = model.predict(X_test)

fig = plt.figure()
ax = fig.add_subplot(211)
ax.scatter(y_test[:, 0], y_test[:, 1], c=y[idx2])
ax.set_xlim((0., 1.))
ax.set_ylim((0., 1.))

ax1 = fig.add_subplot(212)


ax1.scatter(result[:, 0], result[:, 1], c=y[idx2])
ax1.set_xlim((0., 1.))
ax1.set_ylim((0., 1.))
plt.tight_layout()
plt.show()
