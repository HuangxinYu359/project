#使用numpy实现boston房价预测
from sklearn.datasets import load_boston

#step1数据加载
data = load_boston()
x = data['data']
y = data['target']
#print(x,y)

#step2.数据规范化
x_ = (x-x.mean(axis = 0))/x.std(axis=0)
