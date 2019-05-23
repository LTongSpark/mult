# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class tfidf(object):
    def __init__(self):
        self.news = fetch_20newsgroups(subset="all")
    def tf_idf(self):
        # 随机25%做测试数据，75%做训练集
        x_train, x_test, y_train, y_test = train_test_split(self.news.data, self.news.target, test_size=0.25, random_state=33)

        print(x_train)
        tfidf_vec = TfidfVectorizer(analyzer="word" ,stop_words="english")
        x_train = tfidf_vec.fit_transform(x_train)
        x_test = tfidf_vec.transform(x_test)

        # 进行朴素贝叶斯算法的预测
        mlt = MultinomialNB(alpha=1.0)
        mlt.fit(x_train ,y_train)
        y_predict = mlt.predict(x_test)
        print("预测文章的类别为：" ,y_predict)
        # 得出准确率
        print("准确率为：", mlt.score(x_test, y_test))

        print("每个类别的精确率和召回率：", classification_report(y_test, y_predict, target_names=self.news.target_names))


        # #得到语料库中所有不重复词
        # print( "得到语料库中所有不重复词",tfidf_vec.get_feature_names())
        # print("得到停用词" ,tfidf_vec.get_sto4erp_words())
        #
        # #得到每个单词对应的id值
        # print("得到每个单词对应的id值",tfidf_vec.vocabulary_)




if __name__ == '__main__':
    tf = tfidf().tf_idf()
    print(tf)