# -*- coding: utf-8 -*-
import tensorflow as tf

a = 3
w = tf.Variable([[2.3,5.6]])
x = tf.Variable([[2.0],[1.0]])

y = tf.matmul(w,x)

random = tf.random_normal([2,4] ,mean=-1,stddev=4)
shuff = tf.random_shuffle(random)

state = tf.Variable(initial_value=0)
new_value = tf.add(state,1)
update = tf.assign(state,new_value)

init = tf.global_variables_initializer()

#变量

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
out = tf.multiply(input1,input2)


#保存模型

saver = tf.train.Saver()
with tf.Session() as sess:

    #path = saver.save(sess,'C://tensorflow//model//test')

    print(sess.run([out] ,feed_dict={input1:[7.],input2:[8.]}))

    sess.run(init)
    print(sess.run(shuff))
    print("state",sess.run(state))
    print(sess.run(new_value))
    for _ in range(3):
        sess.run(update)
        print("updata",sess.run(update))
