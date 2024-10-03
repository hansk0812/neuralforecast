import numpy as np
from matplotlib import pyplot as plt
x = np.arange(0, 100, 0.1)

y = np.sin(x) * 10
plt.plot(x, y)
plt.show()
