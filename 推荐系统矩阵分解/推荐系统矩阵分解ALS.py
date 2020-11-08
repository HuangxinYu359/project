#数据集来源：https://www.kaggle.com/jneupane12/movielens/download
#baseline论文：http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.476.4158&rep=rep1&type=pdf
#surprise文档：https://surprise.readthedocs.io/en/stable/
from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly
from surprise import accuracy
from surprise.model_selection import KFold

#数据读取
reader = Reader(line_format='user item rating timestamp',sep = ',',skip_lines=1)
data = Dataset.load_from_file('./ratings.csv',reader = reader)
train_set = data.build_full_trainset()

#ALS优化，优化方式可以选其他的（'SGD'）
#设置user、item的正则化项
bsl_options = {'method':'als','n_epochs':5,'reg_u':12,'reg_i':5}
model = BaselineOnly(bsl_options=bsl_options)

#k折交叉验证
kf = KFold(n_splits=5)
for trainset,testset in kf.split(data):
    model.fit(trainset)
    pred = model.test(testset)
    #计算RMSE
    accuracy.rmse(pred)

uid= str(300)
iid = str(180)

#输出uid对iid 的预测结果
pred = model.predict(uid,iid,r_ui=4,verbose=True)