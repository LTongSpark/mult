# -*- coding: utf-8 -*-
from nltk.corpus import brown
from nltk import sent_tokenize,word_tokenize,pos_tag
import seaborn as sns

print(brown.words())
print(brown.tagged_words())
sens = "Machine learning is the science of getting computers to act without being explicitly programmed. In the past decade, machine learning has given us self-driving cars, practical speech recognition, effective web search, and a vastly improved understanding of the human genome. Machine learning is so pervasive today that you probably use it dozens of times a day without knowing it. Many researchers also think it is the best way to make progress towards human-level AI. In this class, you will learn about the most effective machine learning techniques, and gain practice implementing them and getting them to work for yourself. More importantly, you'll learn about not only the theoretical underpinnings of learning, but also gain the practical know-how needed to quickly and powerfully apply these techniques to new problems. Finally, you'll learn about some of Silicon Valley's best practices in innovation as it pertains to machine learning and AI."
#sens = "2019年未来论坛·深圳技术峰会在深圳盛大开幕，深圳市副市长王立新在峰会上表示，“我们每年拿出1/3的财政科技专项资金用于基础研究。今年我们投入在基础研究方面会超过40个亿。”深圳始终把创新和人才作为城市的主导战略，近年来相继出台了一系列的人才政策，吸引了来自海内外各路各路英才，为创新驱动发展提供了源头活水。在境外人才引进政策方面，王立新宣布，来粤港澳大湾区工作的短缺人才也将享受15%的个人所得税减免优惠。"

#nltk不能处理中文数据
#将文章转换成句子的组合
sents = sent_tokenize(sens)
#将文章转换为词的组合
words = word_tokenize(sens)
#有词性
tagged= pos_tag(words)
print(sents)
print(words)
print(tagged)