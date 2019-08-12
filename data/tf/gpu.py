#-*-encoding:utf-8-*-
import tensorflow as tf
x =3
y =4
z =2

with tf.device('/gpu:0'):
    a = tf.multiply(x,x)
    b = tf.multiply(a,y)

with tf.device('/gpu:1'):
    c = tf.add(y,z)

sess  = tf.Session(config = tf.ConfigProto(allow_soft_placement =True ,log_device_placement =True))
d = tf.add(b,c)
print(sess.run(d))
sess.close()
