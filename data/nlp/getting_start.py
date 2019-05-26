# -*- coding: utf-8 -*-
import jieba
import jieba.analyse

# stopword = {}.fromkeys(["的" ,'等','合适','，','。','和','、'])
sens = "故宫的著名景点包括乾清宫、太和殿和午门等。其中乾清宫非常精美，午门是紫禁城的正门，午门居中向阳。"


# jieba.suggest_freq('乾清宫' ,True)
# list = ' '.join([i for i in jieba.lcut(sens,cut_all=False) if i not in stopword ])
jieba.analyse.set_stop_words("../stopwords.txt")
print(jieba.analyse.extract_tags(sens,topK=20 ,withWeight=True))
print(list)