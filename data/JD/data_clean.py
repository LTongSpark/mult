# -*- coding: utf-8 -*-
import pandas as pd

class JD(object):
    def __init__(self):
        self.user_path = 'D:/mr/data/JData_User.csv'
        self.df_month2 ='D:/mr/data/JData_Action_201602.csv'
        self.df_month3 ='D:/mr/data/JData_Action_201603.csv'
        self.df_month4 ='D:/mr/data/JData_Action_201604.csv'
    '''
    首先检查JData_User中的用户和JData_Action中的用户是否一致
    保证行为数据中的所产生的行为均由用户数据中的用户产生（但是可能存在用户在行为数据中无行为）

    思路：利用pd.Merge连接sku 和 Action中的sku, 观察Action中的数据是否减少 Example:
    '''
    def user_action_check(self):
        df_user = pd.read_csv(self.user_path, encoding='gbk')
        df_sku = df_user['user_id']
        # df_month2 = pd.read_csv(self.df_month2, encoding='gbk')
        # print('Is action of Feb. from User file? ', len(df_month2) == len(pd.merge(df_sku, df_month2)))
        # df_month3 = pd.read_csv(self.df_month3, encoding='gbk')
        # print('Is action of Mar. from User file? ', len(df_month3) == len(pd.merge(df_sku, df_month3)))
        # df_month4 = pd.read_csv(self.df_month3, encoding='gbk')
        # print('Is action of Apr. from User file? ', len(df_month4) == len(pd.merge(df_sku, df_month4)))

        df_user['age'] = df_user['age'].apply(self.tranAge())
        print(df_user.groupby(df_user['age']).count())
        df_user.to_csv('data\JData_User.csv', index=None)

    '''
    检查是否有重复记录
    除去各个数据文件中完全重复的记录,可能解释是重复数据是有意义的，比如用户同时购买多件商品，同时添加多个数量的商品到购物车等...
    '''

    def deduplicate(self,filepath, filename, newpath):
        df_file = pd.read_csv(filepath,encoding="gbk")
        before = df_file.shape[0]
        #删除重复的数据
        df_file.drop_duplicates(inplace=True)
        after= df_file.shape[0]

        n_dup = before-after
        print(filename + "重复的数据有" + str(n_dup))
        if n_dup !=0:
            df_file.to_csv(newpath,index=None)
        else:
            print("没有重复的记录")

        #年龄区间的处理
    df_user = pd.read_csv('data\JData_User.csv', encoding='gbk')
    def tranAge(x):
        if x == u'15岁以下':
            x = '1'
        elif x == u'16-25岁':
            x = '2'
        elif x == u'26-35岁':
            x = '3'
        elif x == u'36-45岁':
            x = '4'
        elif x == u'46-55岁':
            x = '5'
        elif x == u'56岁以上':
            x = '6'
        return x

    def run(self):
        print(self.user_action_check())

if __name__ == '__main__':
    JD().run()