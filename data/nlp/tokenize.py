# -*- coding: utf-8 -*-
import nltk.data
import jieba
from nltk.stem.porter import PorterStemmer

sens = "Machine learning is the science of getting computers to act without being explicitly programmed. In the past decade, machine learning has given us self-driving cars, practical speech recognition, effective web search, and a vastly improved understanding of the human genome. Machine learning is so pervasive today that you probably use it dozens of times a day without knowing it. Many researchers also think it is the best way to make progress towards human-level AI. In this class, you will learn about the most effective machine learning techniques, and gain practice implementing them and getting them to work for yourself. More importantly, you'll learn about not only the theoretical underpinnings of learning, but also gain the practical know-how needed to quickly and powerfully apply these techniques to new problems. Finally, you'll learn about some of Silicon Valley's best practices in innovation as it pertains to machine learning and AI."
to = nltk.data.load('tokenizers/punkt/english.pickle')

print(to.tokenize(sens))

#词性归一化

'''
    stemming 词干提取 去掉ing和ed之类的
    lemmatization：转成单数
'''

port = PorterStemmer()
print(port.stem("maxium"))

print(port.stem("playing"))

#安装wordnet
from nltk.stem import WordNetLemmatizer

word = WordNetLemmatizer()
print(word.lemmatize("dogs"))





