#-*-encoding:utf-8-*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.ensemble import GradientBoostingClassifier
#加载数据
data = datasets.load_iris()
#选择训练数据和目标值
X = data['data']
y = data['target']
print(data.feature_names)

#声明算法
gbdt = GradientBoostingClassifier()
#进行特征的选择
gbdt.fit(X, y)
print(gbdt.feature_importances_)

#找出对应的下标值
argsort = gbdt.feature_importances_.argsort()[::-1]
print(type(argsort.tolist()))
#plt.figure(figsize=(9,12))
plt.bar(np.arange(4),gbdt.feature_importances_[argsort])

plt.xticks(np.arange(4) ,[data.feature_names[i] for i in argsort.tolist()])
plt.show()


