# -*- coding: utf-8 -*-
import pandas as pd
import logging
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
data = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
#print(data)
logging.INFO
# 处理数据 找出特征值和目标值
x = data[["pclass", 'age', 'sex']]
y = data["survived"]
# 缺失值的处理
x['age'].fillna(x['age'].mean(), inplace=True)

# 分割数据集到训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
# 进行处理（特征工程）特征-》类别-》one_hot编码
#print(x_train)
dict1 = DictVectorizer(sparse=False)
x_train = dict1.fit_transform(x_train.to_dict("records"))
x_test = dict1.fit_transform(x_test.to_dict("records"))

c_param_range = [0.01, 0.1, 1, 10, 100]
content_list = []
for c_param in c_param_range:
    best_list = dict()
    print("-" * 10)
    print("c_param_range", c_param)
    print("-" * 10)
    lr = LogisticRegression(solver='liblinear')
    lr.fit(x_train, y_train)
    y_predict = lr.predict(x_test)
    print("逻辑回归准确率：", lr.score(x_test, y_test))
    print("召回率", classification_report(y_test, y_predict, labels=[0, 1], target_names=["dead", "nodead"]))
    best_list["flag"] = c_param
    best_list["num"] = lr.score(x_test, y_test)
    content_list.append(best_list)
    print(content_list)
    print(max(content_list,key=lambda x:x['num'])['flag'])
    print(list((max(content_list, key=lambda x: x["num"])).values())[0])
best = max(content_list,key=lambda x:x['num'])['flag']


print(best)

#print(pd.DataFrame({'x_text':x_test},index=[0]))