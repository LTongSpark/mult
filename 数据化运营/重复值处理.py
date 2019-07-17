#-*-encoding:utf-8-*-
import pandas as pd  # 导入pandas库

# 生成重复数据
data1, data2, data3, data4 = ['a', 3], ['b', 2], ['a', 3], ['c', 2]
df = pd.DataFrame([data1, data2, data3, data4], columns=['col1', 'col2'])
print(df)

# 判断重复数据
isDuplicated = df.duplicated()  # 判断重复数据记录
print(isDuplicated)  # 打印输出

# 删除重复值
print(df.drop_duplicates())  # 删除数据记录中所有列值相同的记录
print(df.drop_duplicates(['col1']))  # 删除数据记录中col1值相同的记录
print(df.drop_duplicates(['col2']))  # 删除数据记录中col2值相同的记录
print(df.drop_duplicates(['col1', 'col2']))  # 除数据记录中指定列（col1/col2）值相同的记录