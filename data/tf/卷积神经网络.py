#-*-encoding:utf-8-*-
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
X = tf.placeholder(tf.float64,[None,784])
y = tf.placeholder(tf.float64 ,[None,10])

#卷积核，在卷积神经中是变量
#变量生成方法
def gen_v(shape):
    return tf.Variable(initial_value=tf.random_normal(shape = shape ,dtype = tf.float64 ,stddev =0.1) ,dtype = tf.float64)

#定义方法 完成卷积
def conv(input_data,filter_):
    return tf.nn.conv2d(input_data, filter_, strides = [1,1,1,1] ,padding ="SAME")
#定义方法，完成池化

def pool(input_data):
    return tf.nn.max_pool(value = input_data, ksize = [1,2,2,1] ,strides=[1,2,2,1] ,padding='SAME')

'''
第一层卷积
'''
#因为上面是none ，不确定所以为-1
input_data1 = tf.reshape(X ,shape = [-1,28,28,1])
#偏差
b1 = gen_v(shape=[64])
#卷积核
filter1 = gen_v(shape = [3,3,1,64])
conv1 = conv(input_data1 ,filter1) + b1
#池化
pool1 = pool(conv1)
#激活函数
activer1 = tf.nn.relu(pool1)

'''
第二层卷积
'''

#偏差
b2 = gen_v(shape=[64])
#卷积核
filter2 = gen_v(shape = [3,3,64,64])
conv2 = conv(activer1 ,filter2) + b2
#池化
pool2 = pool(conv2)
#激活函数
activer2 = tf.nn.sigmoid(pool2)

'''
全连接层
1024个连接，1024个方程，1024个神经元
'''

fc_w = gen_v(shape = [7*7*64,1024])
fc_b = gen_v(shape = [1024])
conn = tf.matmul(tf.reshape(activer2, shape = [-1,7*7*64]),fc_w) + fc_b

'''
dropout 防止过拟合
'''
kb = tf.placeholder(tf.float64)
dropout = tf.nn.dropout(conn, keep_prob =kb)

'''
输出层总共有10个类别
'''

out_w = gen_v(shape = [1024,10])
out_b = gen_v(shape=[10])
out = tf.matmul(dropout, out_w) + out_b

#预测的概率
prob = tf.nn.softmax(out)
#交叉熵
cost = tf.reduce_mean(tf.reduce_sum(y*tf.log(1/prob),axis = -1))
cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prob, labels = y))

#定义优化梯度下降
adam = tf.train.AdamOptimizer(learning_rate=0.001)

optimizer = adam.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        c = 0
        for j in range(100):
            X_train ,y_train = mnist.train.next_batch(550)
            optimizer_,cost_ = sess.run(fetches=[optimizer,cost] ,feed_dict = {X: X_train, y: y_train, kb:0.5})
            c += cost_/100
        print("执行次数:%d ,损失函数为：%0.4f"%(i ,c))















