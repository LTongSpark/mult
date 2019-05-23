# -*- coding: utf-8 -*-
import jieba
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import warnings

#对字典数据进行特征化处理
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer

warnings.filterwarnings("ignore" ,category=DeprecationWarning)
dict = DictVectorizer(sparse=False)
# 调用fit_transform
data = dict.fit_transform([{'city': '北京','temperature': 100}, {'city': '上海','temperature':60}, {'city': '深圳','temperature': 30}])

#对文本进行特征化
data_cv = CountVectorizer()
data = data_cv.fit_transform(["人生 苦短，我 喜欢 python", "人生漫长，不用 python"])

con1 = " ".join(list(jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。" ,cut_all=False)))
con2 = " ".join(list(jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",cut_all=False)))
#对中文进行特征化处理
chinese_data = data_cv.fit_transform([con2 ,con1])
#print(chinese_data.toarray())

tf = TfidfVectorizer()
chinese_tf = tf.fit_transform([con1,con2])
#print(chinese_tf.toarray())

#归一化处理
mm = MinMaxScaler(feature_range=(2, 3))#2代表的是下边界，3代表的是上边界，生成的数都是在2~3之间
data_min = mm.fit_transform([[90,2,10,40],[60,4,15,45],[75,3,13,46]])
#print(data)

#标椎化处理
std = StandardScaler()
data_std = std.fit_transform([[ 1., -1., 3.],[ 2., 4., 2.],[ 4., 6., -1.]])
#print(data_std)

#缺失值处理
im = Imputer(missing_values='NaN', strategy='mean', axis=0)
data_im = im.fit_transform([[1, 2], [np.nan, 3], [7, 6],[np.nan ,np.nan]])
#print(data_im)
# 特征选择-删除低方差的特征
var = VarianceThreshold(threshold=1.0)
data_var = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 9, 1, 3]])
#print(data_var)

#主成分分析进行特征降维
pca = PCA(n_components=0.9)
data_pac = pca.fit_transform([[2,8,4,5],[6,3,0,8],[5,4,9,1]])
print(data_pac)



