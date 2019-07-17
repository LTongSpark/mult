#-*-encoding:utf-8-*-
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
#%%
data = np.loadtxt('data6.txt', delimiter='\t')  # 读取数据
#%%
# Z-Score标准化
zscore_scaler = preprocessing.StandardScaler()  # 建立StandardScaler对象
data_scale_1 = zscore_scaler.fit_transform(data)  # StandardScaler标准化处理
#%%
# Max-Min标准化
minmax_scaler = preprocessing.MinMaxScaler()  # 建立MinMaxScaler模型对象
data_scale_2 = minmax_scaler.fit_transform(data)  # MinMaxScaler标准化处理
#%%
# MaxAbsScaler标准化
maxabsscaler_scaler = preprocessing.MaxAbsScaler()  # 建立MaxAbsScaler对象
data_scale_3 = maxabsscaler_scaler.fit_transform(data)  # MaxAbsScaler标准化处理
#%%
# RobustScaler标准化
robustscalerr_scaler = preprocessing.RobustScaler()  # 建立RobustScaler标准化对象
data_scale_4 = robustscalerr_scaler.fit_transform(data)  # RobustScaler标准化标准化处理
#%%
# 展示多网格结果
data_list = [data, data_scale_1, data_scale_2, data_scale_3, data_scale_4]  # 创建数据集列表
color_list = ['black', 'green', 'blue', 'yellow', 'red']  # 创建颜色列表
merker_list = ['o', ',', '+', 's', 'p']  # 创建样式列表
title_list = ['source data', 'zscore_scaler', 'minmax_scaler', 'maxabsscaler_scaler',
              'robustscalerr_scaler']  # 创建标题列表
plt.figure(figsize=(16, 3))
for i, data_single in enumerate(data_list):  # 循环得到索引和每个数值
    plt.subplot(1, 5, i + 1)  # 确定子网格
    plt.scatter(data_single[:, :-1], data_single[:, -1], s=10, marker=merker_list[i],
                c=color_list[i])  # 自网格展示散点图
    plt.title(title_list[i])  # 设置自网格标题
plt.suptitle("raw data and standardized data")  # 设置总标题