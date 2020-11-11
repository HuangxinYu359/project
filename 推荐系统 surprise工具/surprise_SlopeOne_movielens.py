#SlopeOne论文：https://arxiv.org/pdf/cs/0702144.pdf
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import KFold
from surprise import SlopeOne
from surprise import accuracy

reader = Reader(line_format= 'user item rating timestamp',sep = ',',skip_lines=1)
data = Dataset.load_from_file('./ratings.csv',reader = reader)
trainset = data.build_full_trainset()

#使用SlopeOne算法
algo = SlopeOne()
algo.fit(trainset)
#对指定用户和商品进行评分预测
uid =str(200)
iid = str(100)
pred = algo.predict(uid,iid,r_ui=4,verbose=True)