# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

A = [[1, 4, 9, 5, 6]]
B = [[2, 3, 9, 3, 2]]
corr = tf.equal(A,B)
accr = tf.reduce_mean(tf.cast(corr,tf.float32))

with tf.Session() as sess:
    a = tf.cast(corr,tf.float32)
    print(a.eval())
    print(sess.run(accr))