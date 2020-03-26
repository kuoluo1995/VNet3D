import sys

import tensorflow as tf


def get_activation_fn(name):
    if name == 'relu':
        activation_fn = tf.nn.relu
    elif name == 'prelu':
        activation_fn = prelu
    elif name == 'lrelu':
        activation_fn = tf.nn.leaky_relu
    else:
        sys.exit("Invalid activation function")
    return activation_fn


# parametric leaky relu
def prelu(x):
    with tf.variable_scope('prelu'):
        alpha = tf.get_variable('alpha', shape=x.get_shape()[-1], dtype=x.dtype,
                                initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)
