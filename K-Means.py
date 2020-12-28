from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
import time
import unicodedata
import nltk
from io import StringIO

from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler




'''打开数据文本，并读取'''
with open("data/notes1.txt",'r') as text_file:
     # lines = text_file.read().split('\n')
     lines = text_file.read().split(' ')
     print(lines)
text_file.close()

# def leinote():
#     g=[]
#     lines.append(g)
#     return g

'''需要进行聚类的文本集'''
tfidf_vectorizer = TfidfVectorizer(analyzer='word',lowercase=False)
tfidf_matrix = tfidf_vectorizer.fit_transform(lines)
print("*****************输出tfidf_matrix的形状************")
print(tfidf_matrix.shape)
print("n_samples: %d, n_features: %d" % tfidf_matrix.shape)
print("*****************输出tfidf_matrix****************")
print(tfidf_matrix)

features = tfidf_vectorizer.get_feature_names()
print("*****************特征名称******************")
print(tfidf_vectorizer.get_feature_names())

tfidf_weight = tfidf_matrix.toarray()
print("******************转换为权重数组**************")
print(tfidf_weight)

'''使用SOM方法进行聚类'''
# import numpy as np
# from sklearn.datasets import load_digits
# from minisom import MiniSom
# def som(k,x,y,data):
#     # k 为簇数，x,y为神经网络的形状
#     som = MiniSom(x,y,data.shape[1],sigma=1,learning_rate=0.5)
#     som.random_weights_init(np.array(data))
#     som.train_random(np.array(data),2000)
#
#     w = som.get_weights()
#     weight = pd.DataFrame(w.reshape(x*y,data.shape[1]))
#     KM =KMeans(n_clusters=k,n_jobs=-1)
#     model = KM.fit(weight)
#     weight['label'] = model.labels_
#     rs = np.array(weight['label']).reshape((x,y))
#
#     def get_winner(v):
#         l = som.winner(np.array(v))
#         return rs[l]
#
#     return pd.DataFrame(data).apply(get_winner,axis=1)
#     print()
# labels2 = som(9,20,20,tfidf_weight)



'''用于盛放簇内误差平方和的列表'''
distortion = []
k=[]
for i in range(1,16):
    kmeans = KMeans(n_clusters=i, random_state=30)#init='k-means++',
    kmeans.fit(tfidf_matrix)
    distortion.append(kmeans.inertia_)
    print(kmeans.inertia_)
    k.append(i)
    print(i)
plt.plot(range(1,16), distortion)

plt.title('The Elbow Method(sum of the squared errors)')
plt.xlabel('Number of cluster(k)')
plt.ylabel('Distortion(SSE)')
plt.show()


t0 = time.time()
'''真正开始聚类'''
num_clusters = 12
km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40, init='k-means++')

tfidf_matrix = StandardScaler().fit_transform(tfidf_weight)
# km_cluster = OPTICS(min_samples=5, xi=.05, min_cluster_size=.05,max_eps=np.inf,metric='manhattan',p=2,
#                     cluster_method='dbscan',predecessor_correction=True,leaf_size=15)
# km_cluster = Birch(n_clusters=num_clusters,branching_factor=9,threshold=0.1,copy=False)
# from sklearn.mixture import GaussianMixture
# km_cluster = GaussianMixture(n_components=9).fit(tfidf_weight)
# labels = km_cluster.predict(tfidf_weight)

s = km_cluster.fit(tfidf_weight)
print("****************s的值为*************************")
print(s)

print("*****************打印出各个族的中心点**************")
# print(km_cluster.cluster_centers_)
print("******************labels_**********************")
# print(km_cluster.labels_)
# print(km_cluster.labels_.dtype)
# for index, label in enumerate(km_cluster.labels_, 1):
     # print("inex: {}, label: {}".format(index, label))
# 样本距其最近的聚类中心的平方距离之和，用来评判分类的准确度，值越小越好
# k-means的超参数n_clusters可以通过该值来评估
# print("inertia: {}".format(km_cluster.inertia_))
t1 = time.time()
print("clustering time: %.2g sec" % (t1 - t0))

labels = lines
'''
同质性homogeneity：每个群集只包含单个类的成员。
完整性completeness：给定类的所有成员都分配给同一个群集。
两者的调和平均V-measure
兰德系数Adjusted rand index: 值越大意味着聚类结果与真实情况越吻合。
轮廓系数Silhouette Coefficient: 同类别样本距离相近且不同类别样本距离越远，分数越高。

'''
print("Homogeneity值: %0.3f" % metrics.homogeneity_score(labels, km_cluster.labels_))
print("Completeness值: %0.3f" % metrics.completeness_score(labels, km_cluster.labels_))
print("V-measure值: %0.3f" % metrics.v_measure_score(labels, km_cluster.labels_))
print("Adjusted Rand-Index值: %0.3f"
      % metrics.adjusted_rand_score(labels, km_cluster.labels_))
print("Mutual_info_score值：%0.3f" % metrics.adjusted_mutual_info_score(labels, km_cluster.labels_))
print("Silhouette Coefficient值: %0.3f"
      % metrics.silhouette_score(tfidf_matrix, km_cluster.labels_, sample_size=1000))
print("calinski_harabasz_score值: %0.3f" % metrics.calinski_harabasz_score(tfidf_weight,km_cluster.labels_))





'''使用T-SNE算法，准确度比PCA算法高，但是耗时长
   T-SNE降维画出3D图形'''
label1 = [
            '#FFFF00', '#008000', '#0000FF','#8470FF','#FF4500','#98FB98','#A0522D','#FF7F00','#FF6EB4',
            '#8A2BE2',  '#FF3030','#87CEEB','#828282','#7EC0EE','#7CFC00','#7A8B8B','#76EE00','#C0FF3E'
                                                                                                          ]
color=[label1[i] for i in km_cluster.labels_]
# color=[label1[i] for i in labels]
t_0 = time.time()

tsne = TSNE(n_components=3,learning_rate=150,perplexity=30)
# tsne = PCA(n_components=3)
# from sklearn.decomposition import FactorAnalysis
# tsne = FactorAnalysis(n_components = 3)
# from sklearn.decomposition import FastICA
# tsne = FastICA(n_components=3, random_state=12)
# from sklearn import manifold
# tsne = manifold.Isomap( n_components=3)

decomposition_data = tsne.fit_transform(tfidf_weight)   # 转换后的输出
# decomposition_data = tsne.fit(tfidf_matrix).transform(tfidf_matrix)
# print(tsne.explained_variance_ratio_)  #代表降维后各主成分的方差值占各部分方差值的比例
# print(tsne.explained_variance_)        #代表降维后各主成分的方差值
t_1 = time.time()
print("t-SNE: %.2g sec" % (t_1 - t_0))  # 算法用时




'''用训练好的聚类模型反推文档的所属的主题类别'''
label_prediction = km_cluster.predict(tfidf_matrix)
label_prediction = list(label_prediction)

# print("TOP keyword of every clusters:")
# original_space_centroids = tsne.inverse_transform(km_cluster.cluster_centers_)
# order_centroids = original_space_centroids.argsort()[:, ::-1]
# order_centroids = km_cluster.cluster_centers_.argsort()[:, ::-1]
#
# terms = tfidf_vectorizer.get_feature_names()
# for i in range(9):
#     print("Cluster %d   " % (i+1), end='')
#     print("The proportion of documents contained in this cluster is",'%.4f%%' % (int(label_prediction.count(i))/int(len(lines))))
#     print("keywords of the cluster：")
#     for ind in order_centroids[i, :80]:
#         print(' %s,' % terms[ind], end='')
#     print('\n------------------------------------------------------------------------------------------------')


fig = plt.figure(figsize=(10,10))
plt.suptitle("MusicData clustering-3D by K-Means",fontsize=14)
#绘制3D图像
ax = Axes3D(fig)
ax.set(xlabel='TSNE-0', ylabel='TSNE-1',zlabel ='TSNE-2')
ax.scatter(decomposition_data[:, 0], decomposition_data[:, 1], decomposition_data[:, 2],s=0.2,depthshade=True, c=color,marker=',')
ax.view_init(4, -40)  # 初始化视角
# plt.axis('tight')
plt.show()
fig.savefig('./data/3D_sample.eps',format='eps',dpi=1000)






'''使用T-SNE算法，准确度比PCA算法高，但是耗时长
   T-SNE降维画出2D图形'''
# tsne = TSNE(n_components=2,learning_rate=150)

# decomposition_data = tsne.fit_transform(tfidf_weight)
# x = []
# y = []
# for i in decomposition_data:
#     x.append(i[0])
#     y.append(i[1])
# fig, ax = plt.subplots()
# plt.suptitle("MusicData clustering-2D by K-means",fontsize=14)
# plt.scatter(x, y, c= color, s = 0.5, marker=",")
# ax.set(xlabel='TSNE-0', ylabel='TSNE-1')
# ax.grid(False)
# plt.show()
# plt.savefig('./data/2D_sample.png')


# fig = plt.figure(figsize=(10, 10))
# plt.suptitle("MusicData clustering by K-means",fontsize=14)
# ax = plt.axes()
# for i in range(9):
#     plt.scatter(x, y, c= colors_list[i], s = 0.5, marker=",")
# plt.grid(True)
# plt.show()
# plt.savefig('./data/sample.png')



'''
n_clusters: 指定K的值
max_iter: 对于单次初始值计算的最大迭代次数
n_init: 重新选择初始值的次数
init: 制定初始值选择的算法
n_jobs: 进程个数，为-1的时候是指默认跑满CPU
注意，这个对于单个初始值的计算始终只会使用单进程计算，
并行计算只是针对与不同初始值的计算。比如n_init=10，n_jobs=40, 
服务器上面有20个CPU可以开40个进程，最终只会开10个进程
'''
#返回各自文本的所被分配到的类索引
result = km_cluster.fit_predict(tfidf_matrix)
print(result.shape)
print("Predicting result: ", result)
# 第一次运行时将注释打开，项目中会生成doc_cluster.pkl文件，之后运行的时候再注释掉这行就可以使用之前持久化的模型了
# joblib.dump(km_cluster, 'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()



