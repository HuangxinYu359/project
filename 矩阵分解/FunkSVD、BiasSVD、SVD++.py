from surprise import SVD,SVDpp
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import KFold
import pandas as pd
from surprise.model_selection import cross_validate

#数据读取
reader = Reader(line_format='user item rating timestamp',sep = ',',skip_lines=1)
data = Dataset.load_from_file('ratings.csv',reader = reader)
train_set = data.build_full_trainset()

#funkSVD
algo_f = SVD(biased = False)

#BiasSVD
algo_b = SVD(biased = True)

#SVD++
algo_p = SVDpp()

#定义k折交叉验证迭代器，K=3
def eval_model(model):
    kf = KFold(n_splits=3)
    for trainset,testset in kf.split(data):
        #训练并预测
        model.fit(trainset)
        predictions = model.test(testset)
        #计算RMSE
        accuracy.rmse(predictions,verbose=True)
#查看评估结果
print('FunkSVD的RMSE为:')
eval_model(algo_f)

print('BiasSVD的RMSE为:')
eval_model(algo_b)
print('SVD++的RMSE为:')
eval_model(algo_p)


#get_neighbors做电影推荐
uid = str(200)
iid = str(200)
# 输出uid对iid的预测结果
pred = algo_f.predict(uid, iid, r_ui=None, verbose=True)


