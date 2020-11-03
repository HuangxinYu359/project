import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#数据加载
df = pd.read_csv('team_cluster_data.csv',encoding='gbk')
train_x = df[['2019国际排名','2018世界杯排名','2015亚洲杯排名']]
#设置分类个数
GMM = GaussianMixture(n_components=3,covariance_type='full')
GMM.fit(train_x)
predict_GMM = GMM.predict(train_x)
#聚类结果返回到df
df = pd.concat([df,pd.DataFrame(predict_GMM,columns = ['GMM_pred'])],axis =1)
#print(df)

#用kmeans进行测试
SSE = []
for i in range(1,10):
    KM = KMeans(n_clusters=i)
    KM.fit(train_x)
    KM.predict(train_x)
    SSE.append(KM.inertia_)
plt.figure(figsize= (12,9))
plt.plot(range(1,10),SSE)
plt.xlabel('簇数量——聚类的k值')
plt.ylabel('簇的误差平方和SSE')
plt.show()
KM = KMeans(n_clusters=3)
df = pd.concat([df,pd.DataFrame(KM.fit_predict(train_x),columns = ['KM_pred'])],axis =1)
print(df)







