#-*-encoding:utf-8-*-
import numpy as np  # 导入库
import matplotlib.pyplot as plt
#%%
data = np.loadtxt('data5.txt', delimiter='\t')  # 读取数据文件
x = data[:, :-1]  # 切分自变量
correlation_matrix = np.corrcoef(x, rowvar=0)  # 相关性分析
print(correlation_matrix.round(2))  # 打印输出相关性结果
#%%
# 使用Matplotlib展示相关性结果
fig = plt.figure()  # 调用figure创建一个绘图对象
ax = fig.add_subplot(111)  # 设置1个子网格并添加子网格对象
hot_img = ax.matshow(np.abs(correlation_matrix), vmin=0, vmax=1)  # 绘制热力图，值域从0到1
fig.colorbar(hot_img)  # 为热力图生成颜色渐变条
ticks = np.arange(0, 9, 1)  # 生成0-9，步长为1
ax.set_xticks(ticks)  # 生成x轴刻度
ax.set_yticks(ticks)  # 设置y轴刻度
names = ['x' + str(i) for i in range(x.shape[1])]  # 生成坐标轴标签文字
ax.set_xticklabels(names)  # 生成x轴标签
ax.set_yticklabels(names)  # 生成y轴标签