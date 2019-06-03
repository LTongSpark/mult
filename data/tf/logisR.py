# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('/mr/tf/data/', one_hot=True)
training = mnist.train.images
trainlabel = mnist.train.labels
testing = mnist.test.images
testlabel = mnist.test.labels

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32 ,[None,10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y_ = tf.matmul(x,W) + b
lr = tf.nn.softmax(y_)
cost = tf.reduce_mean(-tf.reduce_sum(tf.log(lr) ,reduction_indices=1))

#loss = tf.reduce_mean(tf.square(y-lr))
with tf.name_scope('optm'):
    optm = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

#预测
pred = tf.equal(tf.argmax(lr,1) ,tf.argmax(y,1))

accr = tf.reduce_mean(tf.cast(pred,tf.float32))

init = tf.global_variables_initializer()
training_epochs = 50
batch_size = 50

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        #num_batch = int(mnist.train.num_examples/batch_size)
        for i in range(batch_size):
            batch_xs ,batch_ys = mnist.train.next_batch(batch_size)
            cost_val ,_ = sess.run([cost,optm] ,feed_dict={x:batch_xs ,y:batch_ys})
            feeds = {x: batch_xs, y: batch_ys}
            avg_cost += sess.run(cost, feed_dict=feeds) / batch_size
        if epoch % 5 == 0:
            feeds_train = {x: batch_xs, y: batch_ys}
            feeds_test = {x: mnist.test.images, y: mnist.test.labels}
            train_acc = sess.run(accr, feed_dict=feeds_train)
            test_acc = sess.run(accr, feed_dict=feeds_test)
            print("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f"
                  % (epoch, training_epochs, avg_cost, train_acc, test_acc))
        print("DONE")




