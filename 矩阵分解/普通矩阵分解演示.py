import numpy as np

A = np.array([[5,2],[3,6]])
lamda ,U = np.linalg.eig(A)
print('矩阵A: ')
print(A)
print('特征值: ',lamda)
print('特征向量')
print(U)

#A = U lamda U逆
UI = np.linalg.inv(U)
temp = np.dot(U,[[3,0],[0,8]])
temp = np.dot(temp,UI)

print(temp)
