#-*-encoding:utf-8-*-
import pandas as pd
import jieba  # 结巴分词
from sklearn.feature_extraction.text import TfidfVectorizer  # 基于TF-IDF的词频转向量库
#%%
# 分词函数
def jieba_cut(string):
    return list(jieba.cut(string)) # 精确模式分词
#%%
# 读取自然语言文件和停用词
with open('text.txt', encoding='utf8') as fn1, open('stop_words.txt', encoding='utf8') as fn2:
    string_lines = fn1.read()
    stop_words = fn2.read()
string_lines = string_lines.split('\n')
stop_words = stop_words.split('\n')
#%%
# 中文分词
seg_list = list(map(jieba_cut,string_lines)) # 存储所有分词结果
for i in range(3):  # 打印输出第一行的前5条数据
    print(seg_list[1][i])
#%%
# word to vector
vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=jieba_cut)  # 创建词向量模型
vector_value = vectorizer.fit_transform(string_lines).toarray()  # 将文本数据转换为向量空间模型
vector = vectorizer.get_feature_names()  # 获得词向量
vector_pd = pd.DataFrame(vector_value, columns=vector)  # 创建用于展示的数据框
print(vector_pd.head(1))  # 打印输出第一条数据