#random进行随机选取
import pandas as pd
import numpy as np
import random
from sklearn import datasets

#主函数
def myself_split(x,y,test_size):
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    row = len(x)
    test_row_num = int(row * test_size)
    test_index = random.sample(range(0,row),test_row_num)
    test_x = x.loc[test_index]
    test_y = y.loc[test_index]
    all_index = [i for i in range(row)]
    for i in test_index:
        all_index.remove(i)
    train_index  =all_index
    train_x = x.loc[train_index]
    train_y = y.loc[train_index]
    return train_x,test_x,train_y,test_y

#测试
iris = datasets.load_iris()
#print(iris)
x = iris['data']
y = iris['target']
train_x,test_x,train_y,test_y = myself_split(x,y,0.3)
print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)
#print(train_x.shape(),test_x.shape(),train_y.shape(),test_y.shape())
