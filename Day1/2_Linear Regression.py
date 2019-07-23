import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = X * W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

for count in range(1001):
    cost_value, W_value, b_value, train_value = session.run([cost, W, b, train], feed_dict={X:[8,7,5,3,2], Y:[96,84,60,36,24]})
    if count % 10 == 0:
        print(count, cost_value, W_value, b_value)
