# -*- coding: utf-8 -*-
import jieba
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
import jieba.analyse
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim
from sklearn.model_selection import train_test_split
class news_C(object):
    def __init__(self):
        self.path = "D:/mr/val.txt"
        self.stop_word = "./stopwords.txt"
    def stopword(self):
        stop_words = set()
        with open(self.stop_word, "r", encoding="utf-8") as f:
            for word in f.read():
                stop_words.add(word)
        return stop_words
    def run(self):
        def_news = pd.read_table(self.path,names=['category','theme','URL','content'],encoding='utf-8')
        def_news.dropna()
        content = def_news["content"].tolist()
        clean_word = []
        all_word = []
        for line in content:
            s1_list = [i for i in jieba.cut(line,cut_all=False) if len(i) >1 and i != '\r\n' and i not in self.stopword()]
            for word in s1_list:
                all_word.append(word)
            clean_word.append(s1_list)

        df_content = pd.DataFrame({'clean_word': clean_word})

        df_all_word = pd.DataFrame({'all_word':all_word})
        words_count = df_all_word.groupby(by=['all_word'])["all_word"].agg({'count':np.size})
        words_count = words_count.reset_index().sort_values(by=["count"], ascending=False)
        #画图
        matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)

        wordcloud = WordCloud(font_path="./simhei.ttf", background_color="white", max_font_size=80)
        word_frequence = {x[0]: x[1] for x in words_count.head(10).values}
        wordcloud = wordcloud.fit_words(word_frequence)
        plt.imshow(wordcloud)


        #分析句子前5的单词
        index = 3
        print(def_news['content'][index])
        content_S_str = "".join(clean_word[index])
        print("  ".join(jieba.analyse.extract_tags(content_S_str, topK=5, withWeight=False)))

        # 做映射，相当于词袋 LDA主题模型
        #格式要求  list of list  分好词的结果
        dictionary = corpora.Dictionary(clean_word)
        corpus = [dictionary.doc2bow(sentence) for sentence in clean_word]

        lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)  # 类似Kmeans自己指定K值
        # 一号分类结果

        for topic in lda.print_topics(num_topics=20, num_words=5):
            print(topic[1])

        #进行分词的数据
        df_train = pd.DataFrame({'contents_clean': clean_word, 'label': def_news['category']})
        label = {'汽车':1,'时尚':2}
        df_train['label'] = df_train['label'].map(label)
        x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'], df_train['label'],random_state=1)
        words = []
        for line_index in range(len(x_train)):
            try:
                words.append(' '.join(x_train[line_index]))
            except:
                print(line_index,x_train)
        vec = CountVectorizer(analyzer='word', max_features=4000, lowercase=False)
        vec.fit(words)
        #直接贝叶斯
        classifier = MultinomialNB()
        classifier.fit(vec.transform(words), y_train)

        test_words = []
        for line_index in range(len(x_test)):
            try:
                test_words.append(' '.join(x_test[line_index]))
            except:
                print(line_index)
        classifier.score(vec.transform(test_words), y_test)

        #tf-idf加朴素贝叶斯
        vectorizer = TfidfVectorizer(analyzer='word', max_features=4000, lowercase=False)
        vectorizer.fit(words)
        classifier = MultinomialNB()
        classifier.fit(vectorizer.transform(words), y_train)
        classifier.score(vec.transform(test_words), y_test)
        return None

if __name__ == '__main__':
    print(news_C().run())