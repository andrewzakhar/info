from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


import numpy as np

np.random.seed(11)
N = 6000
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

y = np.sum(x, axis=1)
clf = DecisionTreeRegressor()
clf.fit(x, y)
print(clf.feature_importances_)