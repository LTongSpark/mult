# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import warnings

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
warnings.filterwarnings("ignore")

#定义超参数

learning_rate = 5
ephocs = 10
batch_size = 100

#输入图片为28 * 28 像素784
x = tf.placeholder(tf.float32,[None,784])
#输出为0-9的one-hot编码
y = tf.placeholder(tf.float32,[None,10])

#3定义参数w和b
W1 = tf.Variable(tf.random_normal([784,300],stddev=0.03) ,name='W1')
b1 = tf.Variable(tf.random_normal([300]),name='b1')

tf.get_variable

W2 = tf.Variable(tf.random_normal([784,300],stddev=0.03) ,name='W2')
b2 = tf.Variable(tf.random_normal([300]),name='b2')

#构造隐层网络
'''
z=wx+b
h=relu(z)
'''
hidden_out = tf.add(tf.matmul(x,W1),b1)
hidden_out = tf.nn.relu(hidden_out)

#构造输出
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out,W2),b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))
# 创建优化器，确定优化目标
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimizer(cross_entropy)

# init operator
init_op = tf.global_variables_initializer()

# 创建准确率节点
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 创建session
with tf.Session() as sess:
	# 变量初始化
	sess.run(init_op)
	total_batch = int(len(mnist.train.labels) / batch_size)
	for epoch in range(ephocs):
		avg_cost = 0
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
			_, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
			avg_cost += c / total_batch
		print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
