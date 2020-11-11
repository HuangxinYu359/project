from scipy.linalg import svd
import numpy as np

A = np.array([[1,2],[2,3],[7,8]])
p,s,q = svd(A,full_matrices=False)
print('P=',p)
print('S=',s)
print('Q=',q)
