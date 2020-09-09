import numpy as np

x = np.array([[0, 3, 4], [2, 6, 4]])

x_bar_bar = np.linalg.norm(x, axis=1, keepdims=True)
print(x_bar_bar)

x_norm = x / x_bar_bar
print(x_norm)
