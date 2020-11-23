import numpy as np

#初始化网络，设定权重和偏置
def init_netword():
    network = dict()
    network['W1'] = np.array([[0.3,0.3,0.7],[0.3,0.6,0.9]])
    network['b1'] = np.array([1,1,0.7])
    network['W2'] = np.array([[0.2,0.2],[0.3,0.3],[0.7,0.7]])
    network['b2'] = np.array([2,7])
    network['W3'] = np.array([[0.3,0.4],[0.2,0.9]])
    network['b3'] = np.array([0.2,0.3])
    return network

#激活函数sigmiod
def sigmiod(x):
    return 1/(1+np.exp(-x))

#恒等函数，作为输出层的激活函数
def identity_function(x):
    return x

#前向传播
def forword(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = np.dot(x,W1)+b1
    z1 = sigmiod(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmiod(a2)
    a3 = np.dot(z2,W3)+b3
    z3 = sigmiod(a3)
    y = identity_function(z3)
    return y

#初始化网络
network = init_netword()

#设置输入层
x = np.array([2,3])

#前向传播
y = forword(network,x)
print(y)