#-*-encoding:utf-8-*-

import pandas as pd  # 导入pandas库
from sklearn.preprocessing import OneHotEncoder  # 导入库

# 生成数据
df = pd.DataFrame({'id': [3566841, 6541227, 3512441],
                   'sex': ['male', 'Female', 'Female'],
                   'level': ['high', 'low', 'middle'],
                   'score': [1, 2, 3]})
print(df)  # 打印输出原始数据框

# 使用sklearn进行标志转换
# 拆分ID和数据列
id_data = df[['id']]  # 获得ID列
raw_convert_data = df.iloc[:, 1:]  # 指定要转换的列
print(raw_convert_data)
# 将数值型分类向量转换为标志变量
model_enc = OneHotEncoder()  # 建立标志转换模型对象（也称为哑编码对象）
df_new2 = model_enc.fit_transform(raw_convert_data).toarray()  # 标志转换
# 合并数据
df_all = pd.concat((id_data, pd.DataFrame(df_new2)), axis=1)  # 重新组合为数据框
print(df_all)  # 打印输出转换后的数据框

# 使用pandas的get_dummies做标志转换
df_new3 = pd.get_dummies(raw_convert_data)
df_all2 = pd.concat((id_data, pd.DataFrame(df_new3)), axis=1)  # 重新组合为数据框
print(df_all2)  # 打印输出转换后的数据框
