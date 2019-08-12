# -*- coding: utf-8 -*-
import warnings
import logging
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.externals import joblib
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor, Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

"""
决策树对泰坦尼克号的生死预测
"""

class algorithm():
    def algor(self):
        warnings.filterwarnings("ignore" ,category=DataConversionWarning)
        warnings.filterwarnings("ignore" ,category=FutureWarning ,module="sklearn" ,lineno=196)
        #获取数据
        self.data =  pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
        logging.INFO
        #处理数据 找出特征值和目标值
        x = self.data[["pclass" ,'age' ,'sex']]
        y = self.data["survived"]
        #缺失值的处理
        x['age'].fillna(x['age'].mean() ,inplace=True)

        #分割数据集到训练集和测试集
        x_train,x_test,y_train,y_test = train_test_split(x,y ,test_size=0.4)
        # 进行处理（特征工程）特征-》类别-》one_hot编码
        #print(x_train)
        dict = DictVectorizer(sparse=False)
        x_train = dict.fit_transform(x_train.to_dict("records"))
        x_test = dict.fit_transform(x_test.to_dict("records"))

        print(type(x_train))
        print(type(y_train))
        # 进行标准化处理
        # std = StandardScaler()
        # x_train = std.fit_transform(x_train)
        # x_test = std.transform(x_test)

        #使用神经网络进行预测
        mlp = MLPClassifier(hidden_layer_sizes=(10,),activation='logistic',alpha=0.1,max_iter=1000)
        mlp.fit(x_train,y_train)
        y_predict = mlp.predict(x_test)
        print('准确率',mlp.score(x_test,y_test))
        print("召回率", classification_report(y_test, y_predict, labels=[0, 1], target_names=["dead", "nodead"]))

        #用决策树进行预测
        dec = DecisionTreeClassifier()
        dec.fit(x_train,y_train)
        # 预测准确率
        print("决策树预测的准确率：", dec.score(x_test, y_test))
        print("*" * 100)

        #随机森林进行预测
        rf = RandomForestClassifier()
        param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}

        # 网格搜索与交叉验证
        gc = GridSearchCV(rf, param_grid=param, cv=StratifiedKFold())
        gc.fit(x_train,y_train)

        print("随机森林准确率 : " , gc.score(x_test ,y_test))
        print("查看选择的参数模型：", gc.best_params_)
        print("查看选择的参数模型：", type(gc.best_params_))

        best = gc.best_estimator_

        print("*" * 100)

        #用逻辑回归进行预测
        lr = LogisticRegression(solver='liblinear')
        lr.fit(x_train ,y_train)
        y_predict = lr.predict(x_test)
        print("逻辑回归准确率：" , lr.score(x_test ,y_test))
        print("召回率" ,classification_report(y_test,y_predict ,labels=[0,1] ,target_names=["dead","nodead"]))
        #print(lr.predict_proba(x_test))
        #print(lr.predict_log_proba(x_test))

        print(pd.DataFrame({"x_text":x_test,'per':lr.predict_proba(x_test)}))

        print("*" * 100)

        #svm  支持向量机
        svm = SVC(kernel="rbf",probability=True)
        svm.fit(x_train,y_train)
        print("支持向量机准确率：", svm.score(x_test, y_test))


        print("*" * 100)

        #梯度提升树
        model = GradientBoostingClassifier(n_estimators=200)
        model.fit(x_train,y_train)
        print("梯度提升树准确率 ： " ,model.score(x_test ,y_test))

        #贝叶斯分类器
        mult = MultinomialNB(alpha=0.1)
        mult.fit(x_train ,y_train)

        print("贝叶斯分类器准确率 ：",mult.score(x_test ,y_test))

        #k近邻算法
        knn = KNeighborsClassifier(leaf_size=30,p=2,n_jobs=2)
        knn.fit(x_train ,y_train)
        print("k近邻算法准确率 ：" ,knn.score(x_test ,y_test))


    def predict_house(self):
        """
            线性回归直接预测房子价格
            :return: None
            """
        #获取数据
        lb = load_boston()
        # 分割数据集到训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
        #进行标准化处理
        # 特征值和目标值是都必须进行标准化处理, 实例化两个标准化API
        stand_x= StandardScaler()
        x_train = stand_x.fit_transform(x_train)
        x_test = stand_x.fit_transform(x_test)

        #目标值
        stand_y = StandardScaler()
        y_train = stand_y.fit_transform(y_train.reshape(-1,1))
        y_test = stand_y.fit_transform(y_test.reshape(-1,1))

        #线性回归
        lr = LinearRegression(n_jobs=5)
        lr.fit(x_train ,y_train)
        print(lr.coef_)
        print(lr.intercept_)
        #保存训练好的模型
        joblib.dump(lr,"./model.pkl")
        lr_predict = stand_y.inverse_transform(lr.predict(x_test))
        print("正规方程测试集里面每个房子的预测价格：", lr_predict)
        print("正规方程的均方误差：", mean_squared_error(stand_y.inverse_transform(y_test), lr_predict))

        print("*" *100)
        #梯度下降去进行房价预测
        sgd = SGDRegressor()
        sgd.fit(x_train, y_train)
        print(sgd.coef_)
        # 预测测试集的房子价格
        y_sgd_predict = stand_y.inverse_transform(sgd.predict(x_test))
        print("梯度下降测试集里面每个房子的预测价格：", y_sgd_predict)
        print("梯度下降的均方误差：", mean_squared_error(stand_y.inverse_transform(y_test), y_sgd_predict))

        print("*" * 100)
        # 岭回归去进行房价预测
        rd = Ridge(alpha=1.0)
        rd.fit(x_train, y_train)
        print(rd.coef_)
        # 预测测试集的房子价格
        y_rd_predict = stand_y.inverse_transform(rd.predict(x_test))
        print("梯度下降测试集里面每个房子的预测价格：", y_rd_predict)
        print("梯度下降的均方误差：", mean_squared_error(stand_y.inverse_transform(y_test), y_rd_predict))


        #预测房子的价格
        # model = joblib.load("./model.pkl")
        # predict = stand_y.inverse_transform(model.predict(x_test))
        #
        # print("保存的模型额结果 ：",predict)

if __name__ == '__main__':
    al =algorithm().algor()
    print(al)
