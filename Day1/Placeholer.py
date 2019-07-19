import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = tf.add(a,b)

session = tf.Session()
print(session.run(adder_node, feed_dict={a:3, b:4.5}))
print(session.run(adder_node, feed_dict={a:[1,5], b:[3,10]}))
