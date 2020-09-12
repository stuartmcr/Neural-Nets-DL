import time
import numpy as np

# L1 (yhat, y) = sum (from i = 0 to m) of the abs value of (y^i - yhat^i)
def L1(yhat, y):
    loss = np.sum(abs(y - yhat))
    return loss


yhat = np.array([0.9, 0.2, 0.1, 0.4, 0.9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat, y)))

# L2 (yhat, y) = sum (from i = 0 to m) of (y^i - yhat^i)^2
def L2(yhat, y):
    loss = np.sum((y - yhat) ** 2)
    return loss


yhat = np.array([0.9, 0.2, 0.1, 0.4, 0.9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L2(yhat, y)))
