# -*- coding: utf-8 -*-

import pandas as pd
df_user = pd.read_csv('D:/mr/data/User_table.csv',header=0)
pd.options.display.float_format = '{:,.3f}'.format  #输出格式设置，保留三位小数
print(df_user.describe())

'''
由上述统计信息发现： 第一行中根据User_id统计发现有105321个用户，
发现有3个用户没有age,sex字段，而且根据浏览、加购、删购、购买等记录却只有105180条记录，说明存在用户无任何交互记录，因此可以删除上述用户。

删除没有age,sex字段的用户
'''
df_user[df_user['age'].isnull()]

delete_list = df_user[df_user['age'].isnull()].index
df_user.drop(delete_list,axis=0,inplace=True)

#删除无交互记录的用户
df_naction = df_user[(df_user['browse_num'].isnull()) & (df_user['addcart_num'].isnull()) & (df_user['delcart_num'].isnull()) & (df_user['buy_num'].isnull()) & (df_user['favor_num'].isnull()) & (df_user['click_num'].isnull())]
df_user.drop(df_naction.index,axis=0,inplace=True)
print (len(df_user))


#统计无购买记录的用户
df_bzero = df_user[df_user['buy_num']==0]
#输出购买数为0的总记录数
print (len(df_bzero))


#删除无购买记录的用户
df_user_list = df_user[df_user['buy_num']==0].index
df_user.drop(df_user_list,axis=0,inplace=True)

#由上表所知，浏览购买转换比和点击购买转换比均值为0.018,0.030，因此这里认为浏览购买转换比和点击购买转换比小于0.0005的用户为惰性用户
bindex = df_user[df_user['buy_browse_ratio']<0.0005].index
df_user.drop(bindex,axis=0,inplace=True)

cindex = df_user[df_user['buy_click_ratio']<0.0005].index
df_user.drop(cindex,axis=0,inplace=True)

print(df_user.describe())

