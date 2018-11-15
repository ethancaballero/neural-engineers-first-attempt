import tensorflow as tf
from tensorflow.python.ops import  rnn, rnn_cell, seq2seq

from utils import get_seq_length, _add_gradient_noise, _position_encoding, _xavier_weight_init, _last_relevant, batch_norm


#from https://github.com/DeNeutoy/act-rte-inference/blob/master/AdaptiveIAAModel.py

class Adaptive_Episodes_Config(object):

  init_scale = 0.05
  learning_rate = 0.001
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  encoder_size = 128
  inference_size = 256
  max_epoch = 4
  max_max_epoch = 3
  keep_prob = 0.8
  lr_decay = 0.8
  batch_size = 32
  vocab_size = 10000
  bidirectional = False

  embedding_size = 300
  embedding_reg = 0.0001
  train_embeddings = True
  use_embeddings = False

  eps = 0.1
  max_computation = 20
  step_penalty = 0.00001

#class AdaptiveIAAModel(object):
class Adaptive_Episodes(object):

    """ Implements Iterative Alternating Attention for Machine Reading
        http://arxiv.org/pdf/1606.02245v3.pdf """

    def __init__(self, config, pretrained_embeddings=None,
                 update_embeddings=True, is_training=False):

        self.config = config

    def gate_mechanism(self, gate_input, scope):
        with tf.variable_scope(scope):

            if self.bidirectional:
                size = 3*2*self.config.encoder_size + self.hidden_size
                out_size = 2*self.config.encoder_size
            else:
                size = 3*self.config.encoder_size + self.hidden_size
                out_size = self.config.encoder_size

            hidden1_w = tf.get_variable("hidden1_w", [size, size])
            hidden1_b = tf.get_variable("hidden1_b", [size])

            hidden2_w = tf.get_variable("hidden2_w", [size, size])
            hidden2_b = tf.get_variable("hidden2_b", [size])

            sigmoid_w = tf.get_variable("sigmoid_w", [size, out_size])
            sigmoid_b = tf.get_variable("sigmoid_b", [out_size])

            if self.config.keep_prob < 1.0 and self.is_training:
                gate_input = tf.nn.dropout(gate_input, self.config.keep_prob)

            hidden1 = tf.nn.relu(tf.matmul(gate_input, hidden1_w) + hidden1_b)

            if self.config.keep_prob < 1.0 and self.is_training:
                hidden1 = tf.nn.dropout(hidden1, self.config.keep_prob)

            hidden2 = tf.nn.relu(tf.matmul(hidden1, hidden2_w) + hidden2_b)

            gate_output = tf.nn.sigmoid(tf.matmul(hidden2, sigmoid_w) + sigmoid_b)

        return gate_output

    def get_attention(self, prev_memory, fact_vec):
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=True, initializer=_xavier_weight_init()):

            W_1 = tf.get_variable("W_1")
            b_1 = tf.get_variable("bias_1")

            W_2 = tf.get_variable("W_2")
            b_2 = tf.get_variable("bias_2")

            features = [fact_vec*prev_memory, tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(1, features)

            attention = tf.matmul(tf.tanh(tf.matmul(feature_vec, W_1) + b_1), W_2) + b_2
            
        return attention

    def _attention_GRU_step(self, rnn_input, h, g):
        """Implement attention GRU as described by https://arxiv.org/abs/1603.01417"""
        with tf.variable_scope("attention_gru", reuse=True, initializer=_xavier_weight_init()):

            Wr = tf.get_variable("Wr")
            Ur = tf.get_variable("Ur")
            br = tf.get_variable("bias_r")

            W = tf.get_variable("W")
            U = tf.get_variable("U")
            bh = tf.get_variable("bias_h")

            r = tf.sigmoid(tf.matmul(rnn_input, Wr) + tf.matmul(h, Ur) + br)
            h_hat = tf.tanh(tf.matmul(rnn_input, W) + r*tf.matmul(h, U) + bh)
            rnn_output = g*h_hat + (1-g)*h

            return rnn_output

    #analogous to inference_step
    def generate_episode(self, batch_mask, prob_compare, prob, counter, episode, fact_vecs, acc_states, counter_int, weight_container, bias_container):    
        """Generate episode by applying attention to current fact vectors through a modified GRU"""
        fact_vecs_t = tf.unpack(tf.transpose(fact_vecs, perm=[1,0,2]))

        '''TRY REPLACING acc_states WITH episode AND SEE WHICH WORKS BETTER'''
        attentions = [tf.squeeze(self.get_attention(acc_states, fv), squeeze_dims=[1]) for fv in fact_vecs_t]

        attentions = tf.transpose(tf.pack(attentions))

        softs = tf.nn.softmax(attentions)
        softs = tf.split(1, self.max_input_len, softs)
        
        gru_outputs = []

        # set initial state to zero
        h = tf.zeros((self.batch_size, self.hidden_size))

        # use attention gru
        for i, fv in enumerate(fact_vecs_t):
            h = self._attention_GRU_step(fv, h, softs[i])
            gru_outputs.append(h)

        # extract gru outputs at proper index according to input_lens
        gru_outputs = tf.pack(gru_outputs)
        gru_outputs = tf.transpose(gru_outputs, perm=[1,0,2])

        #analogous to output, new_state = self.inference_cell(input,state)
        episode = _last_relevant(gru_outputs, self.input_len_placeholder)

        ''' # TARGET_SIDE ATTENTION
        episode = self.generate_episode(prev_memory, fact_vecs, concat_all)
        '''

        p = tf.squeeze(tf.sigmoid(self.shared_linear_layer(episode, 1, True)))

        new_batch_mask = tf.logical_and(tf.less(prob + p,self.one_minus_eps),batch_mask)
        new_float_mask = tf.cast(new_batch_mask, tf.float32)
        prob += p * new_float_mask
        prob_compare += p * tf.cast(batch_mask, tf.float32)

        '''based on github.com/tensorflow/tensorflow/issues/5608#issuecomment-260549420'''
        #untied
        Wt = weight_container.read(counter_int)
        bt = bias_container.read(counter_int)
        #tied
        #Wt = weight_container.read(0)
        #bt = bias_container.read(0)
        counter_int+=1

        def use_remainder():
            remainder = tf.constant(1.0, tf.float32,[self.batch_size]) - prob
            remainder_expanded = tf.expand_dims(remainder,1)
            tiled_remainder = tf.tile(remainder_expanded,[1,self.hidden_size])

            acc_state = tf.nn.relu(tf.matmul(tf.concat(1, [acc_states, episode * tiled_remainder]), Wt) + bt)
            return acc_state

        def normal():
            p_expanded = tf.expand_dims(p * new_float_mask,1)
            tiled_p = tf.tile(p_expanded,[1,self.hidden_size])

            acc_state = tf.nn.relu(tf.matmul(tf.concat(1, [acc_states, episode * tiled_p]), Wt) + bt)
            return acc_state 

        counter += tf.constant(1.0,tf.float32,[self.batch_size]) * new_float_mask
        counter_condition = tf.less(counter,self.N)
        condition = tf.reduce_any(tf.logical_and(new_batch_mask,counter_condition))

        acc_state = tf.cond(condition, normal, use_remainder)

        '''ADD MECHANISM TO INCREASE HALT PROB IF MULTIPLE SIMILAR ATTENTION MASKS IN A ROW;
        would be the difference between consecutive attention masks
        based on this cooment: reddit.com/r/MachineLearning/comments/59sfz8/research_learning_to_reason_with_adaptive/d9bgqxw/'''

        return (new_batch_mask, prob_compare, prob, counter, episode, fact_vecs, acc_state, counter_int, weight_container, bias_container)

    #analogous to do_inference_steps
    def do_generate_episodes(self, prev_memory, fact_vecs, batch_size, hidden_size, max_input_len, input_len_placeholder, max_num_hops, epsilon, weight_container, bias_container):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.max_input_len = max_input_len
        self.input_len_placeholder = input_len_placeholder

        counter_int=tf.constant(0)

        self.shared_linear_layer = tf.make_template('shared_linear_layer', tf.nn.rnn_cell._linear)

        self.one_minus_eps = tf.constant(1.0 - epsilon, tf.float32,[self.batch_size])
        self.N = tf.constant(max_num_hops, tf.float32,[self.batch_size])

        prob = tf.constant(0.0,tf.float32,[self.batch_size], name="prob")
        prob_compare = tf.constant(0.0,tf.float32,[self.batch_size], name="prob_compare")
        counter = tf.constant(0.0, tf.float32,[self.batch_size], name="counter")
        self.counter = tf.constant(0.0, tf.float32,[self.batch_size], name="counter")
        acc_states = tf.zeros_like(prev_memory, tf.float32, name="state_accumulator")
        batch_mask = tf.constant(True, tf.bool,[self.batch_size])

        # While loop stops when this predicate is FALSE.
        # Ie all (probability < 1-eps AND counter < N) are false.

        pred = lambda batch_mask, prob_compare, prob,\
                      counter, prev_memory, fact_vecs, acc_state, counter_int, weight_container, bias_container:\
            tf.reduce_any(
                tf.logical_and(
                    tf.less(prob_compare,self.one_minus_eps),
                    tf.less(counter,self.N)))
                # only stop if all of the batch have passed either threshold

            # Do while loop iterations until predicate above is false.

        _,_,remainders,iterations,_,_,state,_,_,_ = \
            tf.while_loop(pred, self.generate_episode,
            [batch_mask, prob_compare, prob,
             counter, prev_memory, fact_vecs, acc_states, counter_int, weight_container, bias_container])

        return state, remainders, iterations