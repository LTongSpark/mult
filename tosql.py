# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sqlalchemy import create_engine
import warnings
import logging
from sklearn.model_selection import train_test_split, GridSearchCV

# 连接数据库的数据
Host = '127.0.0.1'
Port = '3306'
DataBase = 'spark_home'
UserName = 'root'
PassWord = 'root'
# DB_URI的格式：dialect（mysql/sqlite）+driver://username:password@host:port/database?charset=utf8
DB_URI = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(UserName, PassWord, Host, Port, DataBase)
# 1、创建一个engine引擎
engine = create_engine(DB_URI, echo=False)


warnings.filterwarnings("ignore")
#获取数据
data =  pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
print(data.head())
logging.INFO
#处理数据 找出特征值和目标值
x = data[['row.names',"pclass" ,'age' ,'sex']]
y = data["survived"]
#缺失值的处理
x['age'].fillna(x['age'].mean() ,inplace=True)

#分割数据集到训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y ,test_size=0.4)
# 进行处理（特征工程）特征-》类别-》one_hot编码
print(x_train.head())
print(x_train.loc[:,x_train.columns!='row.names'].head())
#print(x_train[[x_train.columns!='row.names']])
dict = DictVectorizer(sparse=False)
x_train1 = dict.fit_transform(x_train.loc[:,x_train.columns!='row.names'].to_dict("records"))
x_test1 = dict.fit_transform(x_test.loc[:,x_test.columns!='row.names'].to_dict("records"))

#随机森林进行预测
rf = RandomForestClassifier()
param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}

# 网格搜索与交叉验证
gc = GridSearchCV(rf, param_grid=param, cv=5)
gc.fit(x_train1, y_train)

print("随机森林准确率 : ", gc.score(x_test1, y_test))
print("查看选择的参数模型：", gc.best_params_)
print("查看选择的参数模型：", type(gc.best_params_))
print(gc.best_params_.get("max_depth"))
print("*" * 100)


#result = pd.DataFrame({"names":x_test['row.names'] , "fear":y_predict})
#result.to_sql("table" ,index=False ,if_exists = "append" ,con=engine)

#print(result)



