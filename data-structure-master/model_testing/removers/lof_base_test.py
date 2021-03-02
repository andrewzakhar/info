from src.data_removing import ExtendedLOF, DBF, LOF
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from src.common import _distances



np.random.seed(11)
psi = np.random.uniform(0, np.pi/2, 500)
phi = np.random.uniform(0, 2 * np.pi, 500)
x1 = np.cos(psi) * np.cos(phi)
x2 = np.cos(psi) * np.sin(phi)
x3 = np.sin(psi)
x = np.array((x1, x2, x3)).T

distance = _distances(x)
nearest_distances = np.nanmin(distance, axis=1)

dbf_x = DBF().fit_delete(x, nearest_distances)
lof_x = ExtendedLOF(n_neighbors=250).fit_delete(x, nearest_distances)

# plot original data
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(211, projection='3d')

ax.scatter(x1, x2, x3, c='b')
ax.scatter(dbf_x[:, 0], dbf_x[:, 1], dbf_x[:, 2], c='r')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')

ax2 = fig.add_subplot(212, projection='3d')

ax2.scatter(x1, x2, x3, c='b')
ax2.scatter(lof_x[:, 0], lof_x[:, 1], lof_x[:, 2], c='r')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('x3')

# plt.tight_layout()
plt.show()
