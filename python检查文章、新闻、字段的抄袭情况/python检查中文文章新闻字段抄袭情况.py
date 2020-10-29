import pandas as pd
import numpy as np
import jieba
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score,accuracy_score,precision_score
import warnings
warnings.filterwarnings('ignore')
#step1.数据加载
data = pd.read_csv('sqlResult.csv',encoding='gb18030')
#print(data.shape)
#print(data.isnull().sum())

#step2.数据预处理1.删除content、source为空的行
data = data.dropna(axis=0,subset=['content'])
data = data.dropna(axis = 0,subset = ['source'])
#print(data.isnull().sum())

#2.结巴分词+加载停用词
with open('chinese_stopwords.txt',encoding='utf-8') as file:
    stopwords = [i[:-1] for i in file.readlines()]
def word_split(text):
    text = text.replace(' ','').replace('/n','')
    text2 = jieba.cut(text)
    result = ' '.join([t for t in text2 if t not in stopwords])
    return result
print(stopwords)
print(data.iloc[0].content)
print(word_split(data.iloc[0].content))
corpus = list(map(word_split,[str(i) for i in data.content]))

#step3.查看文本的特征，TFIDF
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
countVectorizer = CountVectorizer(encoding='gb18030',min_df=0.015)
tfidfTransformer = TfidfTransformer()
countVectorizer = countVectorizer.fit_transform(corpus)
tfidfTransformer = tfidfTransformer.fit_transform(countVectorizer)
print(tfidfTransformer)

#step4标记是否为新华社文章,切分数据集
label = list(map(lambda i:1 if '新华社' in str(i) else 0,data.source))
#数据集切分
x_train,x_test,y_train,y_test = train_test_split(tfidfTransformer.toarray(),label,test_size=0.2)
#多项式朴素贝叶斯、伯努利贝叶斯分类准确率
model_m = MultinomialNB()
model_m.fit(x_train,y_train)
pred_m = model_m.predict(x_test)
print('MultinomialNB的准确率',accuracy_score(y_test,pred_m))
print('MultinomialNB的精确率',precision_score(y_test,pred_m))
print('MultinomialNB的召回率',recall_score(y_test,pred_m))

#step5.预测文章风格是否和自己一致
prediction = model_m.predict(tfidfTransformer.toarray())
label = np.array(label)
compare = pd.DataFrame({'label':label , 'pred':prediction})
copy_index= compare[(compare['label'] == 0)&(compare['pred']==1)].index
print('可能为copy的新闻条数：',len(copy_index))
#实际为新华社的新闻
xinhuashe_news_index = compare[(compare['label'] == 1)].index

#step6 对tfidf.toarray()进行聚类降维
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
normalizer = Normalizer()
scale_array = normalizer.fit_transform(tfidfTransformer.toarray())
kmeans = KMeans(n_clusters=25)
k_label = kmeans.fit_predict(scale_array)
#对分类创建序号
id_class = {index:class_ for index,class_ in enumerate(k_label)}
#step7创建新华社发布的字典
from collections import defaultdict
class_id = defaultdict(set)
for index,class_ in id_class.items():
    if index in xinhuashe_news_index.tolist():
        class_id[class_].add(index)
#查找相似的文章
from sklearn.metrics.pairwise import cosine_similarity
#只在新华社发布的文章中查找
def find_similar_text(cpindex,top=10):
    dist_dict = {i:cosine_similarity(tfidfTransformer[cpindex],tfidfTransformer[i]) for i in class_id[id_class[cpindex]]}
    return sorted(dist_dict.items(),key = lambda  x:x[1][0],reverse=True)[:top]

#step8.如查找某一篇是否为copy的,如果是抄袭的，查看可能抄袭的文章
cpindex = 2447
print('是否在新华社',cpindex in xinhuashe_news_index)
print('是否在copy_news',cpindex in copy_index)
similar_list = find_similar_text(cpindex)
print(similar_list)

print('怀疑抄袭：\n',data.iloc[cpindex].content)
#找一篇相似的原文
similar2 = similar_list[0][0]
print('相似的原文:\n',data.iloc[similar2].content)



