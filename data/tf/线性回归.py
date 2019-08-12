#-*-encoding:utf-8-*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x_train = np.linspace(0,12 ,20).reshape(-1,1)
y_train = np.linspace(3,6,20) + np.random.randn(20)*0.4
# plt.scatter(x_train, y_train)
# plt.show()

lg = LinearRegression()
lg.fit(x_train, y_train)
print(lg.intercept_ ,lg.coef_)


#x是数据
X = tf.placeholder(dtype = tf.float32 ,shape=[20,1])
y = tf.placeholder(dtype = tf.float32, shape = [20])

w = tf.Variable(initial_value=tf.random_normal([1,1]))
b = tf.Variable(initial_value = tf.random_normal([1]))

#创建线性模型
#f = wx + b
#方程返回的值就是理论值
pred = tf.matmul(X,w) + b

#创建tensorflow均方误差cost以及梯度下降优化器optimizer
#cost是一个数：均方误差，越小，说明，理论值和观测值越接近说明计算的w和b越准确
cost = tf.reduce_mean((pred - tf.reshape(y,shape=[20,1]))**2)

gd = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
optimizer = gd.minimize(cost)

epoches = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoches):
        sess.run(optimizer,feed_dict={X:x_train ,y:y_train})
        w_,b_ = sess.run(fetches=[w,b])
        print(w_,b_)
