# -*- coding: utf-8 -*-

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import warnings

warnings.filterwarnings("ignore")

#创建一个常量 =》2
const = tf.constant(2.0,name='const')

#创建tf的变量b和c

b = tf.Variable(2.0,name='b')
c = tf.Variable(2.0,dtype=tf.float32 ,name='c')

qwe = tf.placeholder(tf.float32 ,[None,None])

print(qwe)
#创建operaton
d = tf.add(b,c)
#定义初始化操作  当定义一个变量的时候 ，必须进行初始化
init_op = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    try:
        sess.run(init_op)
        print(sess.run(qwe,feed_dict={qwe : [[1]]}))
        # 把程序的图结构写入事件文件, graph:把指定的图写进事件文件当中
        filewriter = tf.summary.FileWriter("./tmp/test/", graph=sess.graph)
        tf.device
    except TypeError:
        print("type error")
    except ValueError:
        print('value error')
    finally:
        print(123)