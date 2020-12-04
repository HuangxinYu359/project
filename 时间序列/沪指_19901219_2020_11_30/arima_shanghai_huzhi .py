#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima_model import ARIMA
warnings.filterwarnings('ignore')
#中文标注正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']

# # step1.数据加载
data = pd.read_csv('shanghai_1990_12_19_to_2020_11_30.csv',encoding = 'gb2312')
data = data[['日期','收盘价']][::-1]

#step2.将时间作为索引,按月数据进行重采样
data.set_index('日期',inplace = True)
data.index = pd.to_datetime(data.index)
data_month = data.resample('M').mean()
print(data_month)

#step3.设置参数范围
parameters = [[qs,ds,ps] for qs in range(5)  for ds in range(2)  for ps in range(5)]

#寻找最优AMIMA模型参数，即AIC最小
best_aic = float('inf')
for parameter in parameters:
    try:
        model = ARIMA(data_month['收盘价'],order = (parameter[0],parameter[1],parameter[2])).fit()
        aic = model.aic
    except ValueError:
        print('参数错误',parameter)
        continue
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = parameter
# 输出最优模型
print('最优模型: ', best_model.summary())

data_month = data_month.reset_index()
data_month2 = pd.DataFrame({'日期':['2020-12-31','2021-1-31','2021-2-28'],'收盘价':[np.NaN,np.NaN,np.NaN]})
data_month2['日期'] =pd.to_datetime(data_month2['日期'])
data_month = pd.concat([data_month,data_month2])
data_month = data_month.set_index('日期')

#step4.差分预测结果放在future
data_month['future'] = best_model.predict(start = 1,end = 363)
#差分结果画图
plt.figure(figsize= (20,8))
data_month['收盘价'].plot(label = '实际指数')
data_month['future'].plot(label = '预测指数',color = 'y')
plt.show()

#step5.预测结果
data_month.iloc[0,1] = 116.99
for i in range(1,data_month.shape[0]):
    data_month.iloc[i,1]=data_month.iloc[i,1]+data_month.iloc[i-1,1]
#预测结果画图
plt.figure(figsize= (20,8))
data_month['收盘价'].plot(label = '实际指数')
data_month['future'].plot(label = '预测指数',color = 'y')
plt.show()







