import numpy as np
import tensorflow as tf


weights_initializer = tf.contrib.layers.xavier_initializer()
bias_initializer = tf.constant_initializer(0.0)


def tensor_summaries(tensor, scope_name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(tensor)
        tf.summary.scalar(scope_name + '/mean', mean)
        tf.summary.scalar(scope_name + '/stdev',
                          tf.sqrt(tf.reduce_sum(tf.square(tensor - mean))))
        tf.summary.scalar(scope_name + '/min', tf.reduce_min(tensor))
        tf.summary.scalar(scope_name + '/max', tf.reduce_max(tensor))
        tf.summary.histogram(scope_name, tensor)


def batch_norm(input_tensor, momentum, epsilon, training, axis=3,
               center=True, scale=True, fused=True):
    normalization = tf.layers.batch_normalization(
        input_tensor, axis=axis, momentum=momentum, epsilon=epsilon,
        center=center, scale=scale, training=training)
    return normalization


def conv_2d(input_tensor, kernel_size, output_dim, scope_name,
            activation=tf.nn.relu, padding='SAME', strides=[1, 1, 1, 1],
            dropout=False, dropout_rate=0.2, batch_norm=False, summaries=True):
    with tf.variable_scope(scope_name):
        kernel_shape = [kernel_size[0], kernel_size[1],
                        input_tensor.get_shape()[3].value, output_dim]
        weights = tf.get_variable('weights', kernel_shape,
                                  initializer=weights_initializer)
        bias = tf.get_variable('bias', [output_dim],
                               initializer=bias_initializer)
        convolution = tf.nn.conv2d(input_tensor, weights,
                                   strides=strides, padding=padding)

        if batch_norm:
            convolution = convolution

        convolution = activation(convolution + bias, name=scope_name)

        if dropout:
            convolution = tf.nn.dropout(dropout_rate)

        if summaries:
            tensor_summaries(weights, '/weights')
            tensor_summaries(bias, '/bias')

    return convolution


def pool_2d(input_tensor, scope_name, pool_size=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME'):
    with tf.variable_scope(scope_name):
        pool = tf.nn.max_pool(input_tensor, ksize=pool_size, strides=strides,
                              padding=padding)
    return pool


def fully_connected(input_tensor, output_dim, scope_name, activation=None,
                    dropout=True, dropout_rate=0.2, summaries=True):
    input_dim = 0
    input_tensor_shape = input_tensor.get_shape()
    if len(input_tensor_shape) == 4:
        d2 = np.int(input_tensor_shape[1])
        d3 = np.int(input_tensor_shape[2])
        d4 = np.int(input_tensor_shape[3])
        input_dim = d2 * d3 * d4
    if len(input_tensor_shape) == 2:
        input_dim = input_tensor_shape[1]

    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights', [input_dim, output_dim],
                                  initializer=weights_initializer)
        bias = tf.get_variable('bias', [output_dim],
                               initializer=bias_initializer)
        input_tensor_reshaped = tf.reshape(input_tensor, [-1, input_dim])
        fc = tf.add(tf.matmul(input_tensor_reshaped, weights), bias,
                        name='logits')


        if activation is not None:
            fc = activation(fc)

        if dropout:
            fc = tf.nn.dropout(fc, dropout_rate, name='dropout')

        if summaries:
            tensor_summaries(weights, '/weights')
            tensor_summaries(bias, '/bias')

    return fc


def set_loss(loss_type, labels, predictions, scope_name):
    with tf.variable_scope(scope_name):
        if loss_type == 'cross_entropy':
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=predictions)
            loss = tf.reduce_mean(loss)
        if loss_type == 'mse':
            loss = tf.losses.mean_squared_error(labels=labels,
                                               predictions=predictions)

        tf.summary.scalar(scope_name + 'loss', loss)
    return loss


def set_optimizer(loss, optimizer_type='adam', learning_rate=1e-4):
    if optimizer_type == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return optimizer



