"""
    LOF anomalies detection example with default parameters

"""

from src.core import LOF
import numpy as np
import matplotlib.pyplot as plt


x = np.random.uniform(0, 1, size=(1000, 2))
x[-10:] = np.random.uniform(1, 2, size=(10, 2))

predictor = LOF(k=20)
c = predictor.fit_predict(x)

plt.scatter(x[:, 0], x[:, 1], c=c)
plt.show()
