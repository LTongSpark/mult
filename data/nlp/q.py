# -*- coding: utf-8 -*-

import jieba.posseg as pseg
import pandas as pd

import sklearn.feature_extraction

data = pd.read_csv()
'''
加载停用词
'''
def stopword():
    stop_words = set()
    with open('D:\mr\stopwords.txt', "r", encoding="utf-8") as f:
        for word in f.readlines():
            stop_words.add(word.strip())
    return stop_words

'''
中文分词
'''
def clean_text(text):
    words = pseg.cut(''.join(text.split('\t')))
    all_list = list()
    allowPOS = ['n','v','j']
    for word,flag in words:
        if flag in allowPOS and len(word) > 1:
            all_list.append(word)
    return ' '.join([word for word in all_list if word not in stopword()])

'''
把处理过的数据变成Wordvec的要求
'''
list(data.clearn_word.apply(lambda text :text.split(' ')))


