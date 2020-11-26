import torch.nn as nn
import numpy as np
import torch

#数据加载
from sklearn.datasets import load_boston

data = load_boston()
x =data['data']
y = data['target']
#print(y)

y = y.reshape(-1,1)

#数据规范化
from sklearn.preprocessing import MinMaxScaler
mm_scale = MinMaxScaler()
x = mm_scale.fit_transform(x)

#切分数据集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
print(y_test)

#构造网络
model =nn.Sequential(
    nn.Linear(13,10),
    nn.ReLU(),
    nn.Linear(10,1)
)

#定义优化器和损失函数
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

#训练
max_epoch = 300
for i in range(max_epoch):
    #前向传播
    y_pred = model(x_train)
    #计算loss
    loss = criterion(y_pred,y_train)
    #梯度清0
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    #权重调整
    optimizer.step()

#测试
output = model(x_test)
predict_list = output.detach().numpy()
print(predict_list)
