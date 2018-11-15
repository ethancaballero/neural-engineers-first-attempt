import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops, math_ops


def get_seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def get_target_length(sequence):
    used = tf.sign(sequence)
    length = tf.reduce_sum(tf.to_float(used), reduction_indices=1)
    return length


'''alrojo'''
def sequence_loss_tensor(logits, targets, weights, num_classes,
                         average_across_timesteps=True,
                         softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).
    faster? ; 3D tensor logit input; flattens and then multiples in one op; so no for loop 
    """
    with ops.name_scope(name, "sequence_loss_by_example", [logits, targets, weights]):
        probs_flat = tf.reshape(logits, [-1, num_classes])
        targets = tf.reshape(targets, [-1])
        if softmax_loss_function is None:
            crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                probs_flat, targets)
        else:
            crossent = softmax_loss_function(probs_flat, targets)
        crossent = crossent * tf.reshape(weights, [-1])
        crossent = tf.reduce_sum(crossent)
        total_size = math_ops.reduce_sum(weights)
        total_size += 1e-12  # to avoid division by zero
        crossent /= total_size
        return crossent


def _add_gradient_noise(t, stddev=1e-3, name=None):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks."""
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


# from https://github.com/domluna/memn2n
def _position_encoding(sentence_size, embedding_size):
    """Position encoding described in section 4.1 in "End to End Memory Networks" (http://arxiv.org/pdf/1503.08895v5.pdf)"""
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

    # TODO fix positional encoding so that it varies according to sentence lengths


def _xavier_weight_init():
    """Xavier initializer for all variables except embeddings as desribed in [1]"""

    def _xavier_initializer(shape, **kwargs):
        eps = np.sqrt(6) / np.sqrt(np.sum(shape))
        out = tf.random_uniform(shape, minval=-eps, maxval=eps)
        return out

    return _xavier_initializer


# from https://danijar.com/variable-sequence-lengths-in-tensorflow/
# used only for custom attention GRU as TF handles this with the sequence length param for normal RNNs
def _last_relevant(output, length):
    """Finds the output at the end of each input"""
    batch_size = int(output.get_shape()[0])
    max_length = int(output.get_shape()[1])
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


# from therne_utils
def _get_dims(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    fan_out = shape[1] if len(shape) == 2 else shape[-1]
    return fan_in, fan_out


# from therne_utils
def batch_norm(x, is_training):
    """ Batch normalization.
    :param x: Tensor
    :param is_training: boolean tf.Variable, true indicates training phase
    :return: batch-normalized tensor
    """
    with tf.variable_scope('BatchNorm'):
        # calculate dimensions (from tf.contrib.layers.batch_norm)
        inputs_shape = x.get_shape()
        axis = list(range(len(inputs_shape) - 1))
        param_shape = inputs_shape[-1:]

        beta = tf.get_variable('beta', param_shape, initializer=tf.constant_initializer(0.))
        gamma = tf.get_variable('gamma', param_shape, initializer=tf.constant_initializer(1.))
        batch_mean, batch_var = tf.nn.moments(x, axis)
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
