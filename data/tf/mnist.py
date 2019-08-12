# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import warnings


a = tf.constant(np.random.randint(0,10,size=(3,2)),dtype=tf.float32)
b = tf.constant(np.random.randint(0,10,size=(2,4)) ,dtype=tf.float32)
c = np.random.randint(0,10 ,size=1)[0]
#声明变量
d = tf.Variable(initial_value=c)

c = tf.matmul(a, b)
mean = tf.reduce_mean(c,reduction_indices=1)

with tf.Session() as sess:
    warnings.filterwarnings("ignore")
    sess.run(tf.global_variables_initializer())
    print(sess.run(d))
    ret = sess.run(c)
    print(type(ret))
    print(ret.mean(axis=1))
    print(sess.run(mean))



