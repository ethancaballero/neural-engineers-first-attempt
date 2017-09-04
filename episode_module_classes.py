import tensorflow as tf

from therne_attn_gru import AttnGRU
from therne_utils import weight, bias, batch_norm


class CustomCell(tf.nn.rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, num_weights):
        self._num_units = num_units
        self._num_weights = num_weights

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                ru = tf.nn.rnn_cell._linear([inputs, state],
                                            2 * self._num_units, True, 1.0)
                ru = tf.nn.sigmoid(ru)
                r, u = tf.split(1, 2, ru)
            with tf.variable_scope("Candidate"):
                lambdas = tf.nn.rnn_cell._linear([inputs, state], self._num_weights, True)
                lambdas = tf.split(1, self._num_weights, tf.nn.softmax(lambdas))

                Ws = tf.get_variable("Ws",
                                     shape=[self._num_weights, inputs.get_shape()[1], self._num_units])
                Ws = [tf.squeeze(i) for i in tf.split(0, self._num_weights, Ws)]

                candidate_inputs = []

                for idx, W in enumerate(Ws):
                    candidate_inputs.append(tf.matmul(inputs, W) * lambdas[idx])

                Wx = tf.add_n(candidate_inputs)

                c = tf.nn.tanh(Wx + tf.nn.rnn_cell._linear([r * state],
                                                           self._num_units, True, scope="second"))
            new_h = u * state + (1 - u) * c
        return new_h, new_h


class therne_AttnGRU:
    """Attention-based Gated Recurrent Unit cell (cf. https://arxiv.org/abs/1603.01417)."""

    def __init__(self, num_units, is_training, bn):
        self._num_units = num_units
        self.is_training = is_training
        self.batch_norm = bn

    def __call__(self, inputs, state, attention, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or 'AttrGRU'):
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset.
                r = tf.nn.sigmoid(self._linear(inputs, state, bias_default=1.0))
            with tf.variable_scope("Candidate"):
                c = tf.nn.tanh(self._linear(inputs, r * state))

            new_h = attention * c + (1 - attention) * state
        return new_h

    def _linear(self, x, h, bias_default=0.0):
        I, D = x.get_shape().as_list()[1], self._num_units
        w = weight('W', [I, D])
        u = weight('U', [D, D])
        b = bias('b', D, bias_default)

        if self.batch_norm:
            with tf.variable_scope('Linear1'):
                x_w = batch_norm(tf.matmul(x, w), is_training=self.is_training)
            with tf.variable_scope('Linear2'):
                h_u = batch_norm(tf.matmul(h, u), is_training=self.is_training)
            return x_w + h_u + b
        else:
            return tf.matmul(x, w) + tf.matmul(h, u) + b


class EpisodeModule:
    """ Inner GRU module in episodic memory that creates episode vector. """

    def __init__(self, num_hidden, question, facts, is_training, bn):
        self.question = question
        self.facts = tf.unpack(tf.transpose(facts, [1, 2, 0]))  # F x [d, N]

        # transposing for attention
        self.question_transposed = tf.transpose(question)
        self.facts_transposed = [tf.transpose(f) for f in self.facts]  # F x [N, d]

        # parameters
        self.w1 = weight('w1', [num_hidden, 4 * num_hidden])
        self.b1 = bias('b1', [num_hidden, 1])
        self.w2 = weight('w2', [1, num_hidden])
        self.b2 = bias('b2', [1, 1])
        self.gru = AttnGRU(num_hidden, is_training, bn)

    @property
    def init_state(self):
        return tf.zeros_like(self.facts_transposed[0])

    def new(self, memory):
        """ Creates new episode vector (will feed into Episodic Memory GRU)
        :param memory: Previous memory vector
        :return: episode vector
        """
        state = self.init_state
        memory = tf.transpose(memory)  # [N, D]

        with tf.variable_scope('AttnGate') as scope:
            for f, f_t in zip(self.facts, self.facts_transposed):
                g = self.attention(f, memory)
                state = self.gru(f_t, state, g)
                scope.reuse_variables()  # share params

        return state

    def attention(self, f, m):
        """ Attention mechanism. For details, see paper.
        :param f: A fact vector [N, D] at timestep
        :param m: Previous memory vector [N, D]
        :return: attention vector at timestep
        """
        with tf.variable_scope('attention'):
            # NOTE THAT instead of L1 norm we used L2
            q = self.question_transposed
            vec = tf.concat(0, [f * q, f * m, tf.abs(f - q), tf.abs(f - m)])  # [4*d, N]

            # attention learning
            l1 = tf.matmul(self.w1, vec) + self.b1  # [N, d]
            l1 = tf.nn.tanh(l1)
            l2 = tf.matmul(self.w2, l1) + self.b2
            l2 = tf.nn.softmax(l2)
            return tf.transpose(l2)

        return att
