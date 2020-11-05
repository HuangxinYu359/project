#使用SimpleTagBased算法对Delicious数据集进行推荐
#原始数据集：https://grouplens.org/datasets/hetrec-2011/
# 数据格式：userID     bookmarkID     tagID     timestamp
import pandas as pd
import warnings
import math
import random
import operator
warnings.filterwarnings('ignore')

file_path = 'user_taggedbookmarks-timestamps.dat'
#采用字典格式，保存user对item的tag，{user：{item1:[tag1,rag2]...}...}
records = {}
#训练集、测试集
train_data = {}
test_data = {}
#用户标签，商品标签
user_tags = dict()
user_items = dict()
tag_items = dict()

#数据加载
def load_data():
    print('数据正在加载中...')
    df = pd.read_csv(file_path,sep = '\t')
    #将df放入设定的字典格式中
    for i in range(len(df)):
    #for i in range(10):
        uid = df['userID'][i]
        iid = df['bookmarkID'][i]
        tag = df['tagID'][i]
        #setdefault将uid设置为字典，iid设置为[]
        records.setdefault(uid,{})
        records[uid].setdefault(iid,[])
        records[uid][iid].append(tag)
    #print(records)
    print('数据集大小为：%d.' %len(df))
    print('设置tag的人数:%d.' %len(records))
    print('数据加载完成\n')

#将数据集拆分为训练集及测试集,ratio为测试集划分比例
def train_test_split(ratio,seed = 100):
    random.seed(seed)
    for u in records.keys():
        for i in records[u].keys():
            #ratio为设置的比例
            if random.random()<ratio:
                test_data.setdefault(u,{})
                test_data[u].setdefault(i,[])
                for t in records[u][i]:
                    test_data[u][i].append(t)
            else:
                train_data.setdefault(u,{})
                train_data[u].setdefault(i,[])
                for t in records[u][i]:
                    train_data[u][i].append(t)
    print("训练集user数为：%d，测试机user数为：%d." % (len(train_data),len(test_data)))

#设置矩阵mat[index,item]，来储存index与item 的关系, = {index:{item:n}},n为样本个数
def addValueToMat(mat,index,item,value = 1):
    if index not in mat:
        mat.setdefault(index,{})
        mat[index].setdefault(item,value)
    else:
        if item not in mat[index]:
            mat[index].setdefault(item,value)
        else:
            mat[index][item] +=value

#使用训练集,初始化user_tags,user_items,tag_items，/user_tags为{user1：{tags1:n}}
#{user1：{tags2:n}}...{user2：{tags1:n}}，{user2：{tags2:n}}....n为样本个数等
# user_items为{user1:{items1:n}}......原理同上
# tag_items为{tag1:{items1:n}}......原理同上
def initStat():
    records = train_data
    for u,items in records.items():
        for i,tags in records[u].items():
            for tag in tags:
                #users和tag的关系矩阵
                addValueToMat(user_tags,u,tag,1)
                #users和item的关系
                addValueToMat(user_items,u,i,1)
                #tag和item的关系
                addValueToMat(tag_items,tag,i,1)
    print('user_tags,user_items,tag_items初始化完成.')

#对某一用户user进行topN推荐
def recommend(user,N):
    recommend_item = dict()
    tagged_items = user_items[user]
    for tag,utn in user_tags[user].items():
        for item,tin in tag_items[tag].items():
            #如果某一user已经给某一item打过标签，则不推荐
            if item in tagged_items:
                continue
            if item not in recommend_item:
                recommend_item[item] = utn * tin
            else:
                recommend_item[item] = recommend_item[item]+utn*tin
    #按value值，从大到小排序
    return sorted(recommend_item.items(), key=operator.itemgetter(1), reverse=True)[0:N]

#使用测试集，计算准确率和召回率
def precisionAndRecall(N):
    hit = 0
    h_recall = 0
    h_precision = 0
    for user,items in test_data.items():
        if user not in train_data:
            continue
        rank = recommend(user,N)
        for item,rui in rank:
            if item in items:
                hit = hit+1
        h_recall = h_recall +len(items)
        h_precision = h_precision+N

    #返回准确率和召回率
    return (hit/(h_precision*1.0)), (hit/(h_recall*1.0))

#使用test_data对推荐结果进行评估
def testRecommend():
    print('推荐结果评估如下：')
    print("%3s %10s %10s" % ('N', "精确率", '召回率'))
    for n in [5,10,20,40,60,80,100]:
        precision,recall = precisionAndRecall(n)
        print("%3d %10.3f%% %10.3f%%" % (n, precision * 100, recall * 100))


load_data()
train_test_split(0.2)
initStat()
testRecommend()






