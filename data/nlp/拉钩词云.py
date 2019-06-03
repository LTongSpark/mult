# -*- coding: utf-8 -*-

import jieba
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re

text = ''
with open('D:/mr/lagou-job1000-ai-details.txt','r',encoding='utf-8') as f:
    text = f.read()
    f.close()

def stopword():
    stop_words = set()
    with open('D:\pychrom\mult\data\stopwords.txt', "r", encoding="utf-8") as f:
        for word in f.read():
            stop_words.add(word)
    return stop_words

# #词云的图片
# aimask = np.array(Image.open(''))
#jieba分词中文
words =' '.join([i for i in jieba.lcut(text,cut_all = False) if i not in stopword()])
print(words)
#英文
englishs = ' '.join([i for i in jieba.lcut(text,cut_all = False) if re.compile('^[a-zA-Z0-9]]+$').match(i)])
print(englishs)

wc = WordCloud(font_path='./nlp/simhei.ttf',  # 设置字体
               background_color="white",  # 背景颜色
               max_words=1000,  # 词云显示的最大词数
               max_font_size=500,  # 字体最大值
               min_font_size=20, #字体最小值
               random_state=42, #随机数
               collocations=False, #避免重复单词
               #mask=aimask,#造型遮盖
               width=1600,height=1200,margin=10, #图像宽高，字间距，需要配合下面的plt.figure(dpi=xx)放缩才有效
              )
wc.generate(englishs)

#显示词云图
plt.figure(dpi=100)#可以放大缩小
plt.imshow(wc,interpolation='catrom',vmax=100)
plt.axis('off')#隐藏坐标
plt.show()



