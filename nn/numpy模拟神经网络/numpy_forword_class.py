import numpy as np
class Network():
    # 初始化网络，设定权重和偏置
    def __init__(self):
        self.network = dict()
        self.network['W1'] = np.array([[0.3,0.3,0.7],[0.3,0.6,0.9]])
        self.network['b1'] = np.array([1,1,0.7])
        self.network['W2'] = np.array([[0.2,0.2],[0.3,0.3],[0.7,0.7]])
        self.network['b2'] = np.array([2,7])
        self.network['W3'] = np.array([[0.3,0.4],[0.2,0.9]])
        self.network['b3'] = np.array([0.2,0.3])

    #激活函数sigmiod
    def sigmiod(self):
        return 1/(1+np.exp(-self))

    #恒等函数，作为输出层的激活函数
    def identity_function(self):
        return self

    #前向传播
    def forword(self,x):
        W1,W2,W3 = self.network['W1'],self.network['W2'],self.network['W3']
        b1,b2,b3 = self.network['b1'],self.network['b2'],self.network['b3']
        a1 = np.dot(x,W1)+b1
        z1 = Network.sigmiod(a1)
        a2 = np.dot(z1,W2)+b2
        z2 = Network.sigmiod(a2)
        a3 = np.dot(z2,W3)+b3
        z3 = Network.sigmiod(a3)
        y = Network.identity_function(z3)
        return y

a = Network()
x = np.array([2,3])
y = a.forword(x)
print(y)
