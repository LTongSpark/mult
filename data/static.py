# -*- coding: utf-8 -*-
'''
静态数据分析逻辑回归模型
'ADASYN', 'RandomOverSampler',下采样数据
'SMOTE', 'BorderlineSMOTE', 'SVMSMOTE', 'SMOTENC'上采样数据
'''
import pandas as pd
from imblearn.over_sampling import SMOTE,RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import pymysql
import warnings
import numpy as np


class static(object):
    '''
    初始化
    '''

    def __init__(self):
        self.host = "10.212.4.34"
        self.port = 3306
        self.pass_word = "root"
        self.user = "Admin@123"
        self.db = "security_info"
        self.mysql_cn = pymysql.connect(host=self.host, port=self.port, user=self.user, passwd=self.pass_word,
                                        db=self.db)
        self.phone_path = "/home/oycm/shuju/static_phoneNum.txt"
        self.flag_path = "/home/oycm/shuju/static_if_black.txt"
        self.pkl = "/home/oycm/modle/static_mode.pkl"
        self.static_result = "/home/oycm/static_result.txt"

    # 解析数据
    def parse_data(self, datafame, flag):
        datafame["c_cust_cert_code"] = datafame["c_cust_cert_code"].str.replace('[a-zA-Z]', '10')
        datafame.replace(to_replace="", value=np.nan, inplace=True)
        datafame.replace(to_replace="NULL", value=np.nan, inplace=True)
        datafame.fillna(value=0, inplace=True)
        # 设置省份
        datafame["province"] = datafame["c_cust_cert_code"].str[0:2]
        # 设置地级市
        datafame["prefecture"] = datafame["c_cust_cert_code"].str[2:4]
        # 设置州县
        datafame["country"] = datafame["c_cust_cert_code"].str[4:6]
        # 把入网时间拆分成不同的时间
        self.parse_time(datafame, datafame["innet_date"])
        # 把关联号码的时间拆分成不同的时间
        self.parse_time(datafame, datafame['else_innet_date'])
        # 设置黑名单
        datafame["hb_flag"] = flag

    # 解析黑名单数据
    def parse_data_black(self):
        data_black = pd.read_sql('select * from ti_ua_black_user_info_d_20190108;', con=self.mysql_cn)
        self.parse_data(data_black, 1)
        return data_black

    # 解析白名单数据
    def parse_data_white(self):
        data_white = pd.read_sql('select * from ti_ua_white_user_info_d_20190108;', con=self.mysql_cn)
        data_white = pd.read_sql('select * from ti_ua_white_user_info_d_20190108 WHERE LENGTH(c_cust_cert_code) =18;',
                                 con=self.mysql_cn)
        self.parse_data(data_white, 2)
        return data_white

    def run(self):
        warnings.filterwarnings("ignore")
        black = self.parse_data_black()
        white = self.parse_data_white()
        data = pd.concat([black, white], ignore_index=True, axis=0)

        # 分割数据集到训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(
            data[['phone_no', 'cust_id', 'user_id', 'c_cust_cert_code',
                  'age', 'c_real_name_flag', 'create_org_id', 'sts',
                  'call_flag', 'else_phone_no', 'else_cust_id', 'else_create_org_id',
                  'else_sts', 'else_1_money', 'else_2_money',
                  'else_3_money', "province", "day", "week", "hour",
                  "second", "month", "else_day", "else_week", "else_second",
                  "else_month", 'prefecture', 'country'
                  ]], data['hb_flag'], test_size=0.25)

        with open(self.phone_path, 'w+') as f:
            for i in x_test["phone_no"].values:
                f.write(str(i) + "\r")

        std = StandardScaler()
        x_train = std.fit_transform(x_train)
        x_test1 = std.fit_transform(x_test)

        # 上采样数据
        oversampler = SMOTE(random_state=0)
        x_train_over, y_train_over = oversampler.fit_resample(x_train, y_train)
        x_train_over = pd.DataFrame(x_train_over)
        y_train_over = pd.DataFrame(y_train_over)

        # 对逻辑回归的参数进行评估
        c_param_range = [0.01, 0.1, 1, 10, 100]
        content_list = []
        for c_param in c_param_range:
            best_list = dict()
            print("-" * 10)
            print("c_param_range", c_param)
            print("-" * 10)
            lr = LogisticRegression(C=c_param, penalty="l1", solver='liblinear')
            lr.fit(x_train_over, y_train_over)
            y_pred_undersample = lr.predict(x_test1)
            print("逻辑回归准确率：", lr.score(x_test1, y_test))
            print("召回率",
                  classification_report(y_test, y_pred_undersample, labels=[1, 2], target_names=["black", 'white']))
            best_list["flag"] = c_param
            best_list["num"] = lr.score(x_test1, y_test)
            content_list.append(best_list)
            best = list((max(content_list, key=lambda x: x["num"])).values())[0]

        # 好的参数组成的逻辑回归算法
        lr = LogisticRegression(C=best, penalty='l1')
        lr.fit(x_train_over, y_train_over)
        y_pred = lr.predict(x_test1)
        joblib.dump(lr, self.pkl)
        print("逻辑回归准确率：", lr.score(x_test1, y_test))
        print("召回率", classification_report(y_test, y_pred, labels=[1, 2], target_names=['black', 'white']))
        with open(self.flag_path, 'w+') as f:
            for i in y_pred:
                f.write(str(i) + "\r")
        # #加载模型
        # model = joblib.load("/home/oycm/modle/static_mode.pkl")
        # y_predict = model.predict(x_test1)
        self.save_result()

        # 保存数据

    def save_result(self):
        with open(self.phone_path, 'r') as f:
            json_list = f.read().replace('\r', '').replace('\n', ',').split(',')

        with open(self.flag_path, 'r') as f:
            scores_list = f.read().replace('\r', '').replace('\n', ',').split(',')

        with open(self.static_result, 'w+') as f:
            for index in zip(scores_list, json_list):
                f.write(index[0] + "," + index[1] + "\r")

    # 解析时间
    def parse_time(self, data, time):
        time = pd.to_datetime(time)
        time_value = pd.DatetimeIndex(time)
        data["day"] = time_value.day
        data["week"] = time_value.week
        data["hour"] = time_value.hour
        data["second"] = time_value.second
        data["month"] = time_value.month
if __name__ == '__main__':
    static().run()
