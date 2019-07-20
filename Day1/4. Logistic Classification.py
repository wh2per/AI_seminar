import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x1 = tf.placeholder(tf.float32, shape=[None])
x2 = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1+ x2*w2 + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

for count in range(1001):
    cost_value, h_value, train_value = session.run([cost, hypothesis, train], feed_dict={x1:[1,2,3,4], x2:[4,5,6,7], Y:[9,12,15,18]})
    if count % 10 == 0:
        print(count, "cost : ", cost_value, "\nPrediction:\n", h_value)
