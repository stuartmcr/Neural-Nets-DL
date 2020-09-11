import numpy as np
import time

# time.time() is absolute, can be used but not necessary (time from 00:00:00 @ 1/1/1970)
# time.perf_counter() is relative, but continuous (think system time)
# time.process_time() is relative, but does not include breaks (think user time)

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

# Personal math for dot product
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

# ----- DOT PRODUCT with FOR LOOP -----
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i] * x2[i]
toc = time.process_time()
print(
    "dot = "
    + str(dot)
    + "\n ----- Computation time (dot loop) = "
    + str(1000 * (toc - tic))
    + "ms"
)

# ----- DOT PRODUCT with NUMPY -----
tic = time.process_time()
dot = np.dot(x1, x2)
toc = time.process_time()
print(
    "dot = "
    + str(dot)
    + "\n ----- Computation time (dot numpy) = "
    + str(1000 * (toc - tic))
    + "ms"
)

# ----- OUTER PRODUCT with FOR LOOP -----
tic = time.process_time()
outer = np.zeros((len(x1), len(x2)))
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i, j] = x1[i] * x2[j]
toc = time.process_time()
print(
    "outer = "
    + str(outer)
    + "\n ----- Computation time (outer loop) = "
    + str(1000 * (toc - tic))
    + "ms"
)

# ----- OUTER PRODUCT with NUMPY -----
tic = time.process_time()
outer = np.outer(x1, x2)
toc = time.process_time()
print(
    "dot = "
    + str(outer)
    + "\n ----- Computation time (outer numpy) = "
    + str(1000 * (toc - tic))
    + "ms"
)

# ----- ELEMENTWISE IMPLEMENTATION with FOR LOOP -----
tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i] * x2[i]
toc = time.process_time()
print(
    "elementwise multiplication = "
    + str(mul)
    + "\n -----Computation time (elem loop)= "
    + str(1000 * (toc - tic))
    + "ms"
)

# ----- ELEMENTWISE IMPLEMENTATION with NUMPY -----
tic = time.process_time()
mul = np.multiply(x1, x2)
toc = time.process_time()
print(
    "elementwise multiplication = "
    + str(mul)
    + "\n ----- Computation time (elem numpy)= "
    + str(1000 * (toc - tic))
    + "ms"
)

# ----- GENERAL DOT with FOR LOOP -----
W = np.random.rand(3, len(x1))
tic = time.process_time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i, j] * x1[j]
toc = time.process_time()
print(
    "gdot = "
    + str(gdot)
    + "\n ----- Computation time (gen dot loop)= "
    + str(1000 * (toc - tic))
    + "ms"
)

# ----- GENERAL DOT with NUMPY -----
W = np.random.rand(3, len(x1))
tic = time.process_time()
dot = np.dot(W, x1)
toc = time.process_time()
print(
    "gdot = "
    + str(dot)
    + "\n ----- Computation time (gen dot numpy) = "
    + str(1000 * (toc - tic))
    + "ms"
)

