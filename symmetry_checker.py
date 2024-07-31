import math
import numpy as np

# T = [[-1,0,1],[-1,0,-1],[0,-1,0]]
T = [[1,-1,0],[1,1,0],[0,0,1]]
TT = np.transpose(T)

c = [1,1,1]

Ti = np.linalg.inv(TT)

M = np.matmul(c, Ti)
print(M)


cuc = [[3.842, 0, 0], [0, 3.842, 0], [0,0,3.842]]

muc = np.matmul(cuc, TT)
print(muc)