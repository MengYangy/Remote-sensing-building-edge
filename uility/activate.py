import tensorflow as tf


def leak_relu(x, alpha=0.1, name=''):
    return tf.nn.leaky_relu(x, alpha=alpha, name=name)
