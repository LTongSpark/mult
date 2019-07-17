# -*- coding: utf-8 -*-

import pandas as pd
path ='C:/Users/Administrator/Desktop/train_data_small.csv'
data = pd.read_csv(path)
print(data.head(2))
from sklearn.decomposition import PCA

data.fillna(0,inplace=True)
x = data.loc[:,data.columns != 'cust_type']
y = data.loc[:,'cust_type']
print(x)
x_1 = pd.concat([x,x,x,x] ,ignore_index=True ,axis=0)
print(x_1)

pca = PCA(n_components=30 ,whiten=True)#whiten进行白话处理
pca.fit(pd.concat([x,x,x,x] ,ignore_index=False ,axis=0))
print(pca.transform(x))
