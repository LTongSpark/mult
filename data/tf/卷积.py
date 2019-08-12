#-*-encoding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import umap
umap.UMAP()

'''
卷积神经网络构造

1.1 卷积，提取特征
1.2 池化，筛选最大的特征
1.3 激活，加入非线性
1.4 dropout防止过拟合
1.5 归一化操作


卷积 tf.nn.conv2d(input =(四维) ,filter =(四维) , strides =[1,1,1,1] ,padding='VALID')
池化tf.nn.max_pool(conv,ksize =[1,2,2,1] ,strides=[1,2,2,1] ,padding='SAME')
激活tf.nn.reul(池化结果)
dropout tf.nn.dropout(激活结果，keep_prob=int)

全连接层，其实就是矩阵运算
    
out输出层 矩阵运算

'''

image = plt.imread('mayun.jfif')
plt.imshow(image)
plt.show()
print(image.shape)
tou = image[50:175 ,200:350]
filter_ = np.full(shape=[10,10] ,fill_value=1/100).reshape(10,10,1,1)
conv = tf.nn.conv2d(input = image.transpose([2,0,1]).reshape(3,375,500,1).astype(np.float32),
                    filter= filter_ ,strides=[1,1,1,1] ,padding='SAME')


with tf.Session() as sess:
    ret = sess.run(conv)
    print(ret.shape)
    #plt.figure(figsize=(12,9))
    img = ret.reshape(3,375,500).transpose([1,2,0]).astype(np.uint8)
    img[50:175 ,200:350] = tou
    plt.imshow(img)
    plt.show()

