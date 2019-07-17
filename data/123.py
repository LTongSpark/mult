# -*- coding: utf-8 -*-
import pandas as pd
import logging
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
print(data)
logging.INFO
# 处理数据 找出特征值和目标值
x = data[["pclass", 'age', 'sex']]
y = data["survived"]
# 缺失值的处理
x['age'].fillna(x['age'].mean(), inplace=True)

# 分割数据集到训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
# 进行处理（特征工程）特征-》类别-》one_hot编码
print(x_train)
dict = DictVectorizer(sparse=False)
x_train = dict.fit_transform(x_train.to_dict("records"))
x_test = dict.fit_transform(x_test.to_dict("records"))

# 用逻辑回归进行预测
lr = LogisticRegression(solver='liblinear')
lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)
print("逻辑回归准确率：", lr.score(x_test, y_test))
print("召回率", classification_report(y_test, y_predict, labels=[0, 1], target_names=["dead", "nodead"]))

print(pd.DataFrame({'x_text':x_test},index=[0]))