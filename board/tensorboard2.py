from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

with tf.name_scope("input") as scope:
    x = tf.placeholder(tf.float32, [None, 784])

with tf.name_scope("weight") as scope:
    W = tf.Variable(tf.zeros([784, 10]))

with tf.name_scope("bias") as scope:
    b = tf.Variable(tf.zeros([10]))

with tf.name_scope("layer1") as scope:
    y = tf.nn.softmax(tf.matmul(x, W) + b)

w_hist = tf.summary.histogram("weight", W)
b_hist = tf.summary.histogram("bias", b)
y_hist = tf.summary.histogram("y", y)

with tf.name_scope("y_") as scope:
    y_ = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope("cost") as scope:
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    cost_sum = tf.summary.scalar("cost", cross_entropy)

with tf.name_scope("train") as scope:
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/home/jjeong/tf_log", sess.graph)

