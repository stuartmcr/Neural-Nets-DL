import numpy as np
import time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

a = (
    9 * 9
    + 2 * 2
    + 5 * 2
    + 0 * 9
    + 0 * 0
    + 7 * 9
    + 5 * 2
    + 0 * 5
    + 0 * 0
    + 0 * 0
    + 9 * 9
    + 2 * 2
    + 5 * 5
    + 0 * 0
    + 0 * 0
)
print("a = " + str(a))

tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i] * x2[i]
toc = time.process_time()
print(
    "dot = "
    + str(dot)
    + "\n ----- Computation time (loop) = "
    + str(1000 * (toc - tic))
    + "ms"
)

tic = time.process_time()
dot = np.dot(x1, x2)
toc = time.process_time()
print(
    "dot = "
    + str(dot)
    + "\n ----- Computation time (numpy) = "
    + str(1000 * (toc - tic))
    + "ms"
)

