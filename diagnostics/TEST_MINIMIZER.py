import numpy as np
from scipy.optimize import minimize_scalar
# Example arrays
y1 = np.array([1, 2, 3, 4, 5])
y2 = np.array([2, 4, 6, 8, 10])
# Define objective function
def objective(a):
    return np.sum(((a * y1) - y2) ** 2)
# Minimize
result = minimize_scalar(objective)
print("Optimal scale factor a =", result.x)
print("Minimized squared error =", result.fun)