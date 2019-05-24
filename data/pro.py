# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
data_train = pd.read_csv("D:/mr/train.csv")
# 看看各性别的获救情况
myfont = matplotlib.font_manager.FontProperties(fname='./simhei.ttf')
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({'男性': Survived_m, '女性': Survived_f})

fig = plt.figure()
fig.set(alpha=0.65)  # 设置图像透明度，无所谓
plt.title(u"根据舱等级和性别的获救情况",fontproperties = myfont)
plt.xticks([])
plt.yticks([])
ax1 = fig.add_subplot(141)
print(data_train.Survived[data_train.Sex == 'male'].value_counts())
data_train.Survived[data_train.Sex == 'male'].value_counts().plot(kind='bar',label="female highclass",color='#FA2479')
ax1.set_xticklabels(["获救", "未获救"], rotation=0,fontproperties = myfont)
ax1.legend(["男性"], loc='best',prop = myfont)
ax1 = fig.add_subplot(142)
print(data_train.Survived[data_train.Sex == 'female'].value_counts())
data_train.Survived[data_train.Sex == 'female'].value_counts().plot(kind='bar',label="female highclass",color='#BB1234')
ax1.set_xticklabels(["获救", "未获救"], rotation=0,fontproperties = myfont)
ax1.legend(["男性"], loc='best',prop = myfont)
plt.show()