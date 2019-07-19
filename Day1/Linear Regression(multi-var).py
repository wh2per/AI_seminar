import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_data = [[10, 10], [9, 10], [8, 7], [3, 2], [6,2]]
y_data = [[1], [1], [1], [0], [0]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight1')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

for count in range(1001):
    cost_value, _ = session.run([cost, train], feed_dict={X:x_data, Y:y_data})
    if count % 10 == 0:
        print(count, "cost : ", cost_value)

    h,c,a = session.run([hypothesis, predicted, accuracy], feed_dict = {X:x_data, Y:y_data})
    print("hypothesis : ", h, "\nCorrect(Y) : ", c, "\nAccuracy: ", a)
