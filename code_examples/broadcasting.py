import numpy as np
import time

# create 3x4 matrix
A = np.array([[56.0, 0.0, 4.4, 68.0], [1.2, 104.0, 52.0, 8.0], [1.8, 135.0, 99.0, 0.9]])
print(A)

# sum columns
cal = A.sum(axis=0)
print(cal)

# divide individual values by column sum
percentage = 100 * A / cal.reshape(1, 4)
print(percentage)

