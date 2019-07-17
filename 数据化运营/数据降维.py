#-*-encoding:utf-8-*-
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import feature_selection
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import PolynomialFeatures as plf
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from gplearn.genetic import SymbolicTransformer
from sklearn import datasets

# 读取数据文件
data = np.loadtxt('data1.txt')  # 读取文本数据文件
x, y = data[:, :-1], data[:, -1]  # 获得输入的x和目标变量y
print(x[:3])  # 打印输出x的前3条记录

# 基于sklearn的feature_selection做特征选择
# 使用SelectPercentile选择特征
selector_1 = feature_selection.SelectPercentile(percentile=30)
sel_features1 = selector_1.fit_transform(x, y)  # 训练并转换数据
print(sel_features1.shape)  # 打印形状
print(sel_features1[:3])  # 打印前3条记录


# 使用VarianceThreshold选择特征
selector_2 = feature_selection.VarianceThreshold(1)
sel_features2 = selector_2.fit_transform(x)  # 训练并转换数据
print(sel_features2.shape)  # 打印形状
print(sel_features2[:3])  # 打印前3条记录


# 使用RFE选择特征
model_svc = SVC(kernel="linear")
selector_3 = feature_selection.RFE(model_svc, 3)
sel_features3 = selector_3.fit_transform(x, y)  # 训练并转换数据
print(sel_features3.shape)  # 打印形状
print(sel_features3[:3])  # 打印前3条记录
#%%
# 使用SelectFromModel选择特征
model_tree = DecisionTreeClassifier(random_state=0)  # 建立分类决策树模型对象
selector_4 = feature_selection.SelectFromModel(model_tree)
sel_features4 = selector_4.fit_transform(x, y)  # 训练并转换数据
print(sel_features4.shape)  # 打印形状
print(sel_features4[:3])  # 打印前3条记录


# 使用sklearn的LDA进行维度转换
model_lda = LDA()  # 建立LDA模型对象
model_lda.fit(x, y)  # 将数据集输入模型并训练
convert_features = model_lda.transform(x)  # 转换数据
print(convert_features.shape)  # 打印形状
print(model_lda.explained_variance_ratio_)  # 获得各成分解释方差占比
print(convert_features[:3])  # 打印前3条记录


# 使用sklearn的GBDT方法组合特征
model_gbdt = GBDT()
model_gbdt.fit(x, y)
conbine_features = model_gbdt.apply(x)[:, :, 0]
print(conbine_features.shape)  # 打印形状
print(conbine_features[0])  # 打印第1条记录


# 使用sklearn的PolynomialFeatures方法组合特征
model_plf = plf(2)
plf_features = model_plf.fit_transform(x)
print(plf_features.shape)  # 打印形状
print(plf_features[0])  # 打印第1条数据


# 使用gplearn的genetic方法组合特征
data = datasets.load_boston() # 加载数据集
x, y = data.data, data.target  # 分割形成x和y
print(x.shape) # 查看x的形状
print(x[0]) # 查看x的第一条数据
model_symbolic = SymbolicTransformer(n_components=5, generations=18,
                                     function_set=(
                                         'add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg',
                                         'inv','max', 'min'),
                                     max_samples=0.9, metric='pearson',
                                     random_state=0, n_jobs=2)
model_symbolic.fit(x, y)  # 训练数据
symbolic_features = model_symbolic.transform(x)  # 转换数据
print(symbolic_features.shape)  # 打印形状
print(symbolic_features[0])  # 打印第1条数据
print(model_symbolic) # 输出公式


#读者可取消注释执行下面的代码段
#%%
'''
# 本段示例代码将输出重复的重复特征
reg_data = np.loadtxt('data5.txt')
x, y = reg_data[:, :-1], reg_data[:, -1]
model_symbolic = SymbolicTransformer(n_components=5, generations=18,
                                     function_set=(
                                         'add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg',
                                         'inv','max', 'min'),
                                     max_samples=0.9, metric='pearson',
                                     random_state=0, n_jobs=2)
model_symbolic.fit(x, y)  # 训练数据
symbolic_features = model_symbolic.transform(x)  # 转换数据
print(symbolic_features.shape)  # 打印形状
print(symbolic_features[0])  # 打印第1条数据
'''