from surprise import Reader,Dataset
from surprise import KNNBasic,KNNWithMeans,KNNWithZScore,KNNBaseline

#数据读取
reader = Reader(line_format='user item rating timestamp' , sep = ',',skip_lines=1)
data = Dataset.load_from_file('ratings_sample.csv',reader = reader)
trainset = data.build_full_trainset()

#KNNBasic
algo1 = KNNBasic(k =30,sim_options={'name':'MSD','user_based':True},verbose=True)
algo1.fit(trainset)
uid = str(196)
iid = str(302)

pred1 = algo1.predict(uid, iid,verbose=True)
#KNNWithMeans
algo2 = KNNWithMeans(k = 30,sim_options={'name':'cosine','user_based':False},verbose = True)
algo2.fit(trainset)
pred2 = algo2.predict(uid, iid,verbose=True)

#KNNWithZScore f
algo3 = KNNWithZScore(k =30,sim_options={'name':'MSD','user_based':True},verbose=True)
algo3.fit(trainset)
pred3 = algo3.predict(uid, iid,verbose=True)
#KNNBaseline
algo4 = KNNBaseline(k =30,sim_options={'name':'MSD','user_based':True},verbose=True)
algo4.fit(trainset)
pred4 = algo4.predict(uid, iid,verbose=True)


