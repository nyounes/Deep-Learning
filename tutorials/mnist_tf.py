from time import time
import numpy as np
import tensorflow as tf
from tensorflow_tools import conv_2d, pool_2d, fully_connected
from tensorflow_tools import set_loss, set_optimizer
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = None
N_EPOCHS = 12
BATCH_SIZE = 128

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 28, 28, 1])

conv1 = conv_2d(x_image, kernel_size=[5, 5], output_dim=32, scope_name='conv1')
pool1 = pool_2d(conv1, scope_name='pool1')
conv2 = conv_2d(pool1, kernel_size=[5, 5], output_dim=64, scope_name='conv2')
pool2 = pool_2d(conv2, scope_name='pool2')
dense1 = fully_connected(pool2, 1024, activation=tf.nn.relu, dropout=True,
                         dropout_rate=keep_prob, scope_name='dense1')
predictions = fully_connected(dense1, 10, dropout=False, scope_name='output')

loss = set_loss('cross_entropy', labels=y_, predictions=predictions,
                scope_name='loss')

optimizer = set_optimizer(loss)

with tf.name_scope('accuracy'):
    correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_, 1))
    correct_predictions = tf.cast(correct_predictions, tf.float32)
    accuracy = tf.reduce_mean(correct_predictions)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    n_samples = len(mnist.train.labels)
    n_iterations = N_EPOCHS * n_samples
    n_batches = np.int(n_samples / BATCH_SIZE) + 1

    print("Training on {0} during {1} epochs of {2} batches".format(
        n_samples, N_EPOCHS, n_batches))

    current_epoch = 1
    current_batch = 1
    start_time = time()
    for i in range(0, n_iterations, BATCH_SIZE):
        if current_batch % np.int(n_batches / 20) == 0:
            print("Epoch: {:1}/{:1}, Batch: {:1}/{:2}, Time: {:.4}".format(
                current_epoch, N_EPOCHS, current_batch, n_batches,
                np.round(time() - start_time)), end='\r', flush=True)
        x_batch = mnist.train.images[i % n_samples: i % n_samples + BATCH_SIZE]
        y_batch = mnist.train.labels[i % n_samples: i % n_samples + BATCH_SIZE]
        feed_dict = {x: x_batch, y_: y_batch, keep_prob: 0.5}
        sess.run(optimizer, feed_dict=feed_dict)
        if (i % n_samples + BATCH_SIZE) > n_samples:
            feed_dict = {x: x_batch, y_: y_batch, keep_prob: 1.0}
            train_accuracy = accuracy.eval(feed_dict=feed_dict)
            print('Epoch {:1}/{:1}, Accuracy {:.2f}, Time: {:.4}'.format(
                current_epoch, N_EPOCHS, train_accuracy,
                np.round(time() - start_time)) + "           ")
            current_epoch += 1
            current_batch = 1
            start_time = time()
        current_batch += 1

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
