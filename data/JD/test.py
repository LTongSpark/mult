# -*- coding: utf-8 -*-
import pandas as pd

df_month2 = pd.read_csv('D:/mr/data/JData_Action_201603.csv',encoding='gbk')
print(df_month2['user_id'].dtype)
# IsDuplicated = df_month2.duplicated()
# df_d=df_month2[IsDuplicated]
# print(df_d.groupby('type').count())               #发现重复数据大多数都是由于浏览（1），或者点击(6)产生

#检查是否存在诸恶时间是在2016-5-15号后的数据

# df_user = pd.read_csv('D:/mr/data/JData_User.csv',encoding='gbk')
# df_user['user_reg_tm']=pd.to_datetime(df_user['user_reg_tm'])
# qwe= df_user.loc[df_user.user_reg_tm  >= '2016-4-15']
# print(qwe['user_id'].dtype)

'''

user_id	sku_id	time	model_id	type	cate	brand
结论：说明用户没有异常操作数据，所以这一批用户不删除

由于注册时间是京东系统错误造成，如果行为数据中没有在4月15号之后的数据的话，那么说明这些用户还是正常用户，并不需要删除。
'''

# df_month = pd.read_csv('data\JData_Action_201604.csv')
# df_month['time'] = pd.to_datetime(df_month['time'])
# df_month.loc[df_month.time >= '2016-4-16']

#年龄区间的处理


