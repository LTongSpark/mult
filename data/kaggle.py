# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import preprocessing


class kaggle(object):
    def __init__(self):
        self.path = "D:/mr/train.csv"
        self.myfont = matplotlib.font_manager.FontProperties(fname='./simhei.ttf')

    def run(self):
        warnings.filterwarnings("ignore")
        train_df1 = pd.read_csv(self.path ,chunksize = 5000 ,skipinitialspace=True)
        train_df = pd.concat(train_df1 ,ignore_index=True)
        #画图
        #self.picture(data_train=train_df)
        self.set_missing_ages(train_df)
        self.set_missing_embar(train_df)
        self.set_missing_cabin(train_df)
        self.set_missing_sex(train_df)
        self.familySize(train_df)
        self.std_df(train_df)
        del train_df['Age']
        del train_df['Fare']

        y = train_df['Survived']
        x = train_df[['Pclass','Sex','SibSp','Parch','Embarked','familysize','age','fare','PassengerId']]
        x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size=0.6)
        # 对逻辑回归的参数进行评估
        c_param_range = [0.01,0.05, 0.1,0.5,1, 10, 100]
        content_list = []
        for c_param in c_param_range:
            best_list = dict()
            # print("-" * 10)
            # print("c_param_range", c_param)
            # print("-" * 10)
            #以后只要自己要用的数据 但是分割出来还是带有用户id只能是float类型的数据
            lr = LogisticRegression(C=c_param, penalty="l1", solver='liblinear')
            lr.fit(x_train, y_train)
            y_pred_undersample = lr.predict(x_test)
            # print("逻辑回归准确率：", lr.score(x_test, y_test))
            # print("召回率",
            #       classification_report(y_test, y_pred_undersample, labels=[1, 2], target_names=["black", 'white']))
            best_list["flag"] = c_param
            best_list["num"] = lr.score(x_test, y_test)
            content_list.append(best_list)
            best = list((max(content_list, key=lambda x: x["num"])).values())[0]

        lr = LogisticRegression(C=best, penalty="l1", solver='liblinear')

        #5折交叉验证
        cross= cross_val_score(lr, x_train, y_train, cv=5)
        print(np.mean(cross))

        '''
        交叉验证保存错误数据，对错误数据进行人工分析
        '''

        # # 分割数据，按照 训练数据:cv数据 = 7:3的比例
        # split_train, split_cv = train_test_split(df, test_size=0.3, random_state=0)
        # train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
        # # 生成模型
        # clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        # clf.fit(train_df.as_matrix()[:, 1:], train_df.as_matrix()[:, 0])
        #
        # # 对cross validation数据进行预测
        #
        # cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
        # predictions = clf.predict(cv_df.as_matrix()[:, 1:])
        #
        # origin_data_train = pd.read_csv("/Users/HanXiaoyang/Titanic_data/Train.csv")
        # bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(
        #     split_cv[predictions != cv_df.as_matrix()[:, 0]]['PassengerId'].values)]
        # bad_cases

        lr.fit(x_train, y_train)
        y_pred_undersample = lr.predict(x_test)
        print("逻辑回归准确率：", lr.score(x_test, y_test))
        print("召回率",
              classification_report(y_test, y_pred_undersample, labels=[1, 2], target_names=["black", 'white']))
        # x_test['result'] = y_pred_undersample.reshape(-1, 1)
        result = pd.DataFrame({'PassengerId':x_test['PassengerId'], 'Survived':y_pred_undersample.astype(np.int32)})
        print(result)

        # 随机森林进行预测
        rf = RandomForestClassifier()
        param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}

        # 网格搜索与交叉验证
        gc = GridSearchCV(rf, param_grid=param, cv=5)
        gc.fit(x_train, y_train)

        print("随机森林准确率 : ", gc.score(x_test, y_test))
        print("查看选择的参数模型：", gc.best_params_)
        print("查看选择的参数模型：", type(gc.best_params_))

        #用最佳参数进行随机森林预测
        param_best =gc.best_params_
        rf = RandomForestClassifier(n_estimators=param_best.get("n_estimators") ,max_depth=param_best.get("max_depth"))

        rf.fit(x_train.astype(float),y_train.astype(float))
        print("随机森林准确率 : ", rf.score(x_test, y_test))



    #缺失值的处理
    '''
    通常遇到缺值的情况，我们会有几种常见的处理方式

    如果缺值的样本占总数比例极高，我们可能就直接舍弃了，作为特征加入的话，可能反倒带入noise，影响最后的结果了
    如果缺值的样本适中，而该属性非连续值特征属性(比如说类目属性)，那就把NaN作为一个新类别，加到类别特征中
    如果缺值的样本适中，而该属性为连续值特征属性，有时候我们会考虑给定一个step(比如这里的age，我们可以考虑每隔2/3岁为一个步长)，
    然后把它离散化，之后把NaN作为一个type加到属性类目中。
    有些情况下，缺失的值个数并不是特别多，那我们也可以试着根据已有的值，拟合一下数据，补充上。
    '''
    # 填补确实的年龄限制
    def set_missing_ages(self ,df):

        df['Age'].fillna(df['Age'].median(), inplace=True)
    #填补登船港口
    def set_missing_embar(self,df):
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        df.loc[df['Embarked']== 'S','Embarked'] = 1
        df.loc[df['Embarked']== 'Q','Embarked'] = 2
        df.loc[df['Embarked']== 'C','Embarked'] = 3

    def set_missing_sex(self,df):
        df.loc[df['Sex']=='male','Sex'] = 1
        df.loc[df['Sex']=='female','Sex'] = 2

    def set_missing_cabin(self,df):
        df.drop(['Cabin','Name','Ticket'],axis=1,inplace=True)

    def familySize(self,df):
        df['familysize'] = df['SibSp'] + df['Parch'] +1

    def std_df(self,df):
        scaler = preprocessing.StandardScaler()
        df['age'] = scaler.fit_transform(df[['Age']])
        df['fare'] = scaler.fit_transform(df[['Fare']])

    def missingdata(self,data):

        total = data.isnull().sum().sort_values(ascending=False)
        percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending=False)
        ms = pd.concat([total ,percent] ,axis=1 ,keys=['total' ,'percent'])
        ms = ms.loc[ms["total"] >0]
        return ms

    def picture(self,data_train):
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数

        plt.subplot2grid((2, 3), (0, 0))  # 在一张大图里分列几个小图
        data_train.Survived.value_counts().plot(kind='bar')  # 柱状图
        plt.title("获救情况 (1为获救)" ,fontproperties = self.myfont)  # 标题
        plt.ylabel("人数" ,fontproperties = self.myfont)

        plt.subplot2grid((2, 3), (0, 1))
        data_train.Pclass.value_counts().plot(kind="bar")
        plt.ylabel("人数",fontproperties = self.myfont)
        plt.title("乘客等级分布",fontproperties = self.myfont)

        plt.subplot2grid((2, 3), (0, 2))
        plt.scatter(data_train.Survived, data_train.Age)
        plt.ylabel("年龄",fontproperties = self.myfont)  # 设定纵坐标名称
        plt.grid(b=True, which='major', axis='y')
        plt.title("按年龄看获救分布 (1为获救)",fontproperties = self.myfont)

        plt.subplot2grid((2, 3), (1, 0), colspan=2)
        data_train.Age[data_train.Pclass == 1].plot(kind='kde')
        data_train.Age[data_train.Pclass == 2].plot(kind='kde')
        data_train.Age[data_train.Pclass == 3].plot(kind='kde')
        plt.xlabel("年龄",fontproperties = self.myfont)  # plots an axis lable
        plt.ylabel("密度",fontproperties = self.myfont)
        plt.title("各等级的乘客年龄分布",fontproperties = self.myfont)
        plt.legend(('头等舱', '2等舱', '3等舱'), loc='best',prop = self.myfont)  # sets our legend for our graph.

        plt.subplot2grid((2, 3), (1, 2))
        data_train.Embarked.value_counts().plot(kind='bar')
        plt.title("各登船口岸上船人数",fontproperties = self.myfont)
        plt.ylabel("人数",fontproperties = self.myfont)
        plt.show()

        # 看看各乘客等级的获救情况
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数

        Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
        Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
        df = pd.DataFrame({'获救': Survived_1, '未获救': Survived_0})
        df.plot(kind='bar', stacked=True)
        plt.title("各乘客等级的获救情况" ,fontproperties = self.myfont)
        plt.xlabel("乘客等级",fontproperties = self.myfont)
        plt.ylabel("人数",fontproperties = self.myfont)
        plt.legend(('获救','未获救'), loc='best', prop=self.myfont)
        plt.show()

        # 看看各性别的获救情况
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数

        Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
        Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
        df = pd.DataFrame({'男性': Survived_m, '女性': Survived_f})
        df.plot(kind='bar', stacked=True)
        plt.title("按性别看获救情况",fontproperties = self.myfont)
        plt.xlabel("性别",fontproperties = self.myfont)
        plt.ylabel("人数",fontproperties = self.myfont)
        plt.legend(('获救', '未获救'), loc='best', prop=self.myfont)
        plt.show()

        #另一种表示方法
        fig = plt.figure()
        fig.set(alpha=0.65)  # 设置图像透明度，无所谓
        plt.title(u"根据舱等级和性别的获救情况", fontproperties=self.myfont)
        plt.xticks([])
        plt.yticks([])
        ax1 = fig.add_subplot(141)
        print(data_train.Survived[data_train.Sex == 'male'].value_counts())
        data_train.Survived[data_train.Sex == 'male'].value_counts().plot(kind='bar', label="female highclass",
                                                                          color='#FA2479')
        ax1.set_xticklabels(["获救", "未获救"], rotation=0, fontproperties=self.myfont)
        ax1.legend(["男性"], loc='best', prop=self.myfont)
        ax1 = fig.add_subplot(142)
        print(data_train.Survived[data_train.Sex == 'female'].value_counts())
        data_train.Survived[data_train.Sex == 'female'].value_counts().plot(kind='bar', label="female highclass",
                                                                            color='#BB1234')
        ax1.set_xticklabels(["获救", "未获救"], rotation=0, fontproperties=self.myfont)
        ax1.legend(["男性"], loc='best', prop=self.myfont)
        plt.show()

        # 然后我们再来看看各种舱级别情况下各性别的获救情况
        fig = plt.figure()
        fig.set(alpha=0.65)  # 设置图像透明度，无所谓
        plt.title(u"根据舱等级和性别的获救情况",fontproperties = self.myfont)
        plt.xticks([])
        ax1 = fig.add_subplot(141)
        data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar',
                                                                                                    label="female highclass",
                                                                                                    color='#FA2479')
        ax1.set_xticklabels(["获救", "未获救"], rotation=0,fontproperties = self.myfont)
        ax1.legend(["女性/高级舱"], loc='best',prop = self.myfont)

        ax2 = fig.add_subplot(142, sharey=ax1)
        data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar',
                                                                                                    label='female, low class',
                                                                                                    color='pink')
        ax2.set_xticklabels(["未获救", "获救"], rotation=0,fontproperties = self.myfont)
        plt.legend(["女性/低级舱"], loc='best',prop = self.myfont)

        ax3 = fig.add_subplot(143, sharey=ax1)
        data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar',
                                                                                                  label='male, high class',
                                                                                                  color='lightblue')
        ax3.set_xticklabels(["未获救", "获救"], rotation=0,fontproperties = self.myfont)
        plt.legend(["男性/高级舱"], loc='best',prop = self.myfont)

        ax4 = fig.add_subplot(144, sharey=ax1)
        data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar',
                                                                                                  label='male low class',
                                                                                                  color='steelblue')
        ax4.set_xticklabels(["未获救", "获救"], rotation=0,fontproperties = self.myfont)
        plt.legend(["男性/低级舱"], loc='best',prop = self.myfont)

        plt.show()

        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数

        Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
        Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
        df = pd.DataFrame({'获救': Survived_1, '未获救': Survived_0})
        df.plot(kind='bar', stacked=True)
        plt.title("各登录港口乘客的获救情况",fontproperties = self.myfont)
        plt.xlabel("登录港口",fontproperties = self.myfont)
        plt.ylabel("人数",fontproperties = self.myfont)
        plt.legend(['获救','未获救'] ,loc='best',prop=self.myfont)

        plt.show()

        #有无Cabin记录影响
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数

        Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
        Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
        df = pd.DataFrame({'有': Survived_cabin, '无': Survived_nocabin})
        df.plot(kind='bar', stacked=True)
        plt.title("按Cabin有无看获救情况",fontproperties = self.myfont)
        plt.xlabel("Cabin有无",fontproperties = self.myfont)
        plt.ylabel("人数",fontproperties = self.myfont)
        plt.legend(['获救', '未获救'], loc='best', prop=self.myfont)
        plt.show()


if __name__ == '__main__':
    kaggle().run()