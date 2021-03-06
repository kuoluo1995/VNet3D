import tensorflow as tf


def get_activation_fn(x, activation_type):
    if activation_type is None:
        return x
    elif activation_type == 'prelu':
        return prelu(x)
    elif activation_type == 'lrelu':
        return tf.nn.leaky_relu(x)
    elif activation_type == 'relu':
        return tf.nn.relu(x)
    else:
        raise Exception("Invalid activation function")


# parametric leaky relu
def prelu(x):
    with tf.variable_scope('prelu'):
        alpha = tf.get_variable('alpha', shape=x.get_shape()[-1], dtype=x.dtype,
                                initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)
