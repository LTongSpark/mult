# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

X = np.array([[1,1],[1,1.1],[0,0],[0,0.1]])
y = np.array([1,1,0,0])
knn.fit(X,y)


data = knn.predict(np.array([[0.1,0.1],[1.1,1.1]]))
print(data)
print(knn.predict_proba(np.array([[0.1,0.1]])))

from sklearn.datasets import load_iris
# 使用加载器读取数据并且存入变量iris
iris = load_iris()

# 查验数据规模
iris.data.shape

# 查看数据说明（这是一个好习惯）
print(iris.DESCR)