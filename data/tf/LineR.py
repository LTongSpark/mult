# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#随机生成1000个点，围绕在y=0.1x + 0.3 的直线周围

num_points = 1000
vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0,0.55)
    y1 = x1 *0.1 + 0.3 + np.random.normal(0.0,0.03)
    vectors_set.append([x1,y1])

#生成一些样本数据
x_data = [ i[0] for i in vectors_set]
y_data = [ i[1] for i in vectors_set]

plt.scatter(x_data,y_data ,c = 'r')
plt.show()

#利用tf来实现逻辑回归
#生成1维的w矩阵，取值是【-1,1】之间的随机数
W = tf.Variable(tf.random_uniform([1] ,-1.0,1.0) ,name='w')
#生成1维的b矩阵，初始值为0
b = tf.Variable(tf.zeros([1]) ,name='b')

#经过计算的出预估值y
y = W * x_data + b
#以预估值y好实际值y_data之间的均方误差作为损失
loss = tf.reduce_mean(tf.square(y-y_data) ,name='loss')
#采用梯度下降法来优化参数  训练额过程就是最小化这个误差值
with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#初始化

init = tf.global_variables_initializer()
bath_size = 20

with tf.Session() as sess:
    sess.run(init)
    # 初始化的W和b是多少
    print("W =", sess.run(W), "b =", sess.run(b), "loss =", sess.run(loss))
    for i in range(bath_size):
        sess.run(optimizer)
        # 输出训练好的W和b
        print( "训练" ,"W =", sess.run(W), "b =", sess.run(b), "loss =", sess.run(loss))

    plt.scatter(x_data,y_data,c='r')
    plt.plot(x_data,sess.run(W)*x_data+sess.run(b))
    plt.show()



