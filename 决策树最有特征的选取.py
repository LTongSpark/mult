# -*- coding: utf-8 -*-

from math import log
import pandas as pd

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
data.columns= ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']

x = data.ix[:,0:4]
y = data['class']
'''
函数说明 创建数据集
return：
dataset 数据集
labels 分类属性
'''
def createDataSet():
    dataSet = [[0, 0, 0, 0,'no'],
               [0,0,0,1,'no'],
               [0,1,0,1,'yes'],
               [0,1,1,0,'yes'],
                [0,0,0,0,'no'],
                [1,0,0,0,'no'],
                [1,0,0,1,'no'],
                [1,1,1,1,'yes'],
                [1,0,1,2,'yes'],
                [1,0,1,2,'yes'],
                [2,0,1,2,'yes'],
                [2,0,1,1,'yes'],
                [2,1,0,1,'yes'],
                [2,1,0,2,'yes'],
                [2,0,0,0,'no']]
    labels = ['不放贷' ,'房贷']
    return dataSet,labels

'''
函数说明  计算给定数据集的经验熵（香农熵）
Parameters: 
dataSet 
Returns: 
shann
'''

def calcShannonEnt(dataSet):
    # 这个函数对计算经验熵和条件熵都是相同的，因为公式其实并无区别。只是条件熵加上了一个条件，数据集变小了而已
    numEntires = len(dataSet)  # 返回数据集的行数
    labelCounts = {}  # 保存每个标签(Label)出现次数的字典
    for featVec in dataSet:  # 对每组特征向量进行统计
        currentLabel = featVec[-1]  # 提取标签(Label)信息
        if currentLabel not in labelCounts.keys():  # 如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # Label计数
    shannonEnt = 0.0  # 经验熵(香农熵)
    for key in labelCounts:  # 计算香农熵
        prob = float(labelCounts[key]) / numEntires  # 选择该标签(Label)的概率
        shannonEnt -= prob * log(prob, 2)  # 利用公式计算计算香农熵
    return shannonEnt

'''
函数说明，按照给定的特征划分数据集
Parameters: 
    dataSet - 待划分的数据集 
    axis - 划分数据集的特征 
    value - 需要返回的特征的值 
Returns: 
    无 
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []                                        #创建返回的数据集列表
    for featVec in dataSet:                             #遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]                #去掉axis特征
            reducedFeatVec.extend(featVec[axis+1:])     #将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet #返回划分后的数据集


""" 
函数说明:选择最优特征 
Parameters: 
    dataSet - 数据集 
Returns: 
    bestFeature - 信息增益最大的(最优)特征的索引值 
Modify: 
    2018-09-12 
"""
def chooseBestFeatureToSplit(dataSet):
    print(len(dataSet))
    numFeatures = len(dataSet[0]) - 1                    #特征数量
    baseEntropy = calcShannonEnt(dataSet)                 #计算数据集的香农熵
    bestInfoGain = 0.0                                  #信息增益
    bestFeature = -1                                    #最优特征的索引值
    for i in range(numFeatures):                         #遍历所有特征
        #获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)                         #创建set集合{},元素不可重复
        newEntropy = 0.0                                  #经验条件熵
        for value in uniqueVals:                         #计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)         #subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))           #计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)     #根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy                     #信息增益
        if (infoGain > bestInfoGain):                             #计算信息增益
            bestInfoGain = infoGain                             #更新信息增益，找到最大的信息增益
            bestFeature = i                                     #记录信息增益最大的特征的索引值
    return bestFeature

if __name__ == '__main__':
    print("最优特征索引值:" + str(chooseBestFeatureToSplit(x)))



