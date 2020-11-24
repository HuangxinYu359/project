#使用numpy实现一个神经网络
import numpy as np

#n为样本大小,d_in为输入层维度，h为隐藏层维度（只有一层隐藏层）d_out为输出层维度
n,d_in,h,d_out = 1,1,1,1

#随机生成数据
x = np.array([100])
y = np.array([200])
#print(x,y)

#随机初始化权重
#输入层到隐藏层（100，50）
w1 = np.array([1.5])
#隐藏层到输出层（50，1）
w2 = np.array([1.5])
#设置学习率
learning_rate = 1e-6

#100次迭代
for i in range(100):
    #前向传播
    temp = np.dot(x,w1)

    y_pred = np.dot(temp,w2)

    #计算前向传播的损失函数
    loss = np.square(y_pred-y).sum()/n
    print(loss)

    #基于loss梯度，反向传播
    grad_y_pred = 2.0*(y_pred-y)
    grad_w2 = np.dot(temp.T,grad_y_pred)
    grad_temp_relu = np.dot(grad_y_pred,w2.T)
    grad_temp = grad_temp_relu.copy()
    grad_w1 = x.T.dot(grad_temp)

    # 更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
