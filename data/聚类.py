# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np


#GMM 聚类算法
class ClassJoin(object):
    def __init__(self):
        self.path = "D:\mr\data.txt"

    def kmean(self):
        data = pd.read_csv(self.path,sep=' ')
        x = data[["calories","sodium","alcohol","cost"]]
        std = StandardScaler()
        a_std = std.fit_transform(x)
        km = KMeans(n_clusters=3)
        colors = np.array(['red', 'green', 'blue', 'yellow'])
        km.fit(a_std)
        data['flag'] = km.labels_
        print(data.sort_values("flag"))
        pd.plotting.scatter_matrix(data, c=colors[data['flag']], alpha=1, figsize=(10,10), s=100)
        #聚类评估：轮廓函数
        from sklearn import metrics
        score_scaled = metrics.silhouette_score(x, data.flag)
        print(score_scaled)

        scores = []
        for k in range(2, 20):
            labels = KMeans(n_clusters=k).fit(x).labels_
            score = metrics.silhouette_score(x, labels)
            scores.append(score)
        print(scores)
        plt.plot(list(range(2, 20)), scores)

    def dbscan(self):
        colors = np.array(['red', 'green', 'blue', 'yellow'])
        from sklearn.cluster import DBSCAN
        data = pd.read_csv(self.path, sep=' ')
        x = data[["calories", "sodium", "alcohol", "cost"]]
        db = DBSCAN(eps=10, min_samples=2).fit(x)
        data['flag'] = db.labels_
        pd.plotting.scatter_matrix(data, c=colors[data['flag']], alpha=1, figsize=(10, 10), s=100)


    def gmm(self):
        data = pd.read_csv(self.path, sep=' ')
        x = data[["calories", "sodium", "alcohol", "cost"]]
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=4).fit(x)
        labels = gmm.predict(x)
        labels
        plt.scatter(x[:1],x[:2], c=labels, s=40, cmp='viridis')



if __name__ == '__main__':
    ClassJoin().gmm()