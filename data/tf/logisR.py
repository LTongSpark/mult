# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('/mr/tf/', one_hot=True)
training = mnist.train.images
trainlabel = mnist.train.labels
testing = mnist.test.images
testlabel = mnist.test.labels

#定义变量占位符
x = tf.placeholder(tf.float64,[None,784])
y = tf.placeholder(tf.float64 ,[None,10])
W = tf.Variable(initial_value=tf.random_normal([784,10],dtype = tf.float64 ),dtype=tf.float64)
b = tf.Variable(initial_value=tf.random_normal([10] ,dtype = tf.float64),dtype=tf.float64)

#预测的数据
#根据数学原理写方程
pred = tf.matmul(x,W) + b
#非真实分布 y 真实的数据

y_ = tf.nn.softmax(pred)
#交叉熵  越小越好
#定义损失函数cost
cost = tf.reduce_mean(tf.reduce_sum(tf.multiply(y,tf.log(1/y_)) ,axis=1))
#定义优化梯度下降
gd = tf.train.GradientDescentOptimizer(0.01)
optimizer = gd.minimize(cost)

#保存模型
saver = tf.train.Saver()

#定义session训练 for循环
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        for j in range(100):
            X_train, y_train = mnist.train.next_batch(550)
            optimizer_ ,cost_ = sess.run(fetches=[optimizer,cost] ,feed_dict={x:X_train, y: y_train})
        print("循环%d次数，损失函数值为%0.2f"%(i,cost_))
    if (i+1)%10 == 0:
        saver.save(sess, save_path ="" ,global_step = i)

'''
计算准确率
'''

with tf.Session() as sess:
    saver.restore(sess, save_path = "")
    X_test, y_test = mnist.test.next_batch(2000)
    #准确率
    pred_ = sess.run(fetches=y_ ,feed_dict={x:X_test})
    pred_ = np.argmax(pred_,axis =1)
    y_test = np.argmax(y_test, axis =1)
    print("算法预测准确率：",(tf.reduce_mean(tf.equal(pred_, y_test))))



