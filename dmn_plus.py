import sys
import time

import pdb
import numpy as np
from copy import deepcopy

import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, array_ops, nn_ops, math_ops
from tensorflow.python.framework import ops
from tensorflow.python.util import nest

from tensorflow.contrib.layers import fully_connected

from rnn_decoder_utils import rnn_decoder_simple
import data_utils
from adaptive_attention import Adaptive_Episodes, Adaptive_Episodes_Config
from utils import get_seq_length, get_target_length, sequence_loss_tensor, _add_gradient_noise, _position_encoding, _xavier_weight_init, _last_relevant, batch_norm


class Config(object):
    """Holds model hyperparams and data information."""

    batch_size = 25

    embed_size = 80
    hidden_size = 80

    max_epochs = 256
    early_stopping = 20

    dropout = 0.9
    batch_norm = 0
    layer_norm = 0
    lr = 0.001
    l2 = 0.001

    cap_grads = False
    max_grad_val = 10
    noisy_grads = False

    word2vec_init = False
    embedding_init = 1.7320508 # root 3

    # set to zero with strong supervision to only train gates
    strong_supervision = False
    beta = 1

    drop_grus = False

    anneal_threshold = 1000
    anneal_by = 1.5

    num_hops = 18
    num_attention_features = 2

    max_allowed_inputs = 130

    num_train = 439


    floatX = np.float32

    babi_id = "1"
    babi_test_id = ""

    train_mode = True

    #how likely you want halts to be
    epsilon = 0.2  # <-- this one eventually learns to not attend so is fine
    #epsilon = 0.5 # <-- this allows it to have the option of easily not attending for a decode timestep

    #how much more do you want longer targets to effect loss function
    #between 0 and 1
    length_weighting = .1


class SharedGRUCell(tf.nn.rnn_cell.GRUCell):
    def __init__(self, num_units, input_size=None, activation=tf.nn.tanh):
        tf.nn.rnn_cell.GRUCell.__init__(self, num_units, input_size, activation)
        self.my_scope = None

    def __call__(self, a, b):
        if self.my_scope == None:
            self.my_scope = tf.get_variable_scope()
        else:
            self.my_scope.reuse_variables()
        return tf.nn.rnn_cell.GRUCell.__call__(self, a, b, self.my_scope)

class DMN_PLUS(object):

    def load_data(self, debug=False):
        """Loads train/valid/test data and sentence encoding"""

        en_train, fr_train, en_dev, fr_dev, en_vocab_path, fr_vocab_path = data_utils.prepare_data('tmp', 40000, 40000)

        self.source_vocab_to_id, self.source_id_to_vocab = data_utils.initialize_vocabulary(en_vocab_path)
        self.target_vocab_to_id, self.target_id_to_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

        source_path = './tmp/train.ids40000.questions'
        target_path = './tmp/train.ids40000.answers'

        if self.config.train_mode:
            source_path = './tmp/train.ids40000.questions'
            target_path = './tmp/train.ids40000.answers'
            sources, targets = data_utils.read_data(source_path, target_path)
        else:
            source_path = './tmp/test.ids40000.questions'
            target_path = './tmp/test.ids40000.answers'
            sources, targets = data_utils.read_data(source_path, target_path)

        self.train, self.valid, self.max_t_len, self.max_input_len, self.max_sen_len = data_utils.pad_length_bucket(sources, targets, self.config)

        source_vocab_path = './tmp/vocab40000.questions'
        target_vocab_path = './tmp/vocab40000.answers'
        self.source_vocab_size = data_utils.get_vocab_size(source_vocab_path)
        self.target_vocab_size = data_utils.get_vocab_size(target_vocab_path)

        self.word_embedding = np.random.uniform(-self.config.embedding_init, self.config.embedding_init, (self.source_vocab_size, self.config.embed_size))

    def add_placeholders(self):
        """add data placeholder to graph"""
        self.target_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_t_len))
        self.input_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_input_len, self.max_sen_len))

        self.target_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))
        self.input_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))

        self.dropout_placeholder = tf.placeholder(tf.float32)

    def add_reused_variables(self):
        """Adds trainable variables which are later reused""" 
        gru_cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
        self.shared_gru_cell_before_dropout = SharedGRUCell(self.config.hidden_size)

        attn_length = 1
        '''^DEFINATELY TRY OUT DIFFERENT LENGTHS'''
        with tf.variable_scope('input/forward', initializer=_xavier_weight_init(), reuse=True):
            self.intra_attention_GRU_cell_fw = tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.AttentionCellWrapper(
                self.shared_gru_cell_before_dropout, attn_length, state_is_tuple=False), input_keep_prob=self.dropout_placeholder, output_keep_prob=self.dropout_placeholder)
                
        with tf.variable_scope('input/backward', initializer=_xavier_weight_init(), reuse=True):
            self.intra_attention_GRU_cell_bw = tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.AttentionCellWrapper(
                self.shared_gru_cell_before_dropout, attn_length, state_is_tuple=False), input_keep_prob=self.dropout_placeholder, output_keep_prob=self.dropout_placeholder)
                
        # apply droput to grus if flag set
        if self.config.drop_grus:
            self.gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, input_keep_prob=self.dropout_placeholder, output_keep_prob=self.dropout_placeholder)
        else:
            self.gru_cell = gru_cell

        with tf.variable_scope("memory/attention", initializer=_xavier_weight_init()):
            b_1 = tf.get_variable("bias_1", (self.config.embed_size,))
            W_1 = tf.get_variable("W_1", (self.config.embed_size*self.config.num_attention_features, self.config.embed_size))

            W_2 = tf.get_variable("W_2", (self.config.embed_size, 1))
            b_2 = tf.get_variable("bias_2", 1)

        with tf.variable_scope("memory/attention_gru", initializer=_xavier_weight_init()):
            Wr = tf.get_variable("Wr", (self.config.embed_size, self.config.hidden_size))
            Ur = tf.get_variable("Ur", (self.config.hidden_size, self.config.hidden_size))
            br = tf.get_variable("bias_r", (1, self.config.hidden_size))

            W = tf.get_variable("W", (self.config.embed_size, self.config.hidden_size))
            U = tf.get_variable("U", (self.config.hidden_size, self.config.hidden_size))
            bh = tf.get_variable("bias_h", (1, self.config.hidden_size))

        with tf.variable_scope("memory/normal_gru", initializer=_xavier_weight_init()):

            Wu = tf.get_variable("Wu", (self.config.embed_size+self.target_vocab_size, self.config.hidden_size))
            Uu = tf.get_variable("Uu", (self.config.hidden_size, self.config.hidden_size))
            bu = tf.get_variable("bias_u", (1, self.config.hidden_size))

            Wr = tf.get_variable("Wr", (self.config.embed_size+self.target_vocab_size, self.config.hidden_size))
            Ur = tf.get_variable("Ur", (self.config.hidden_size, self.config.hidden_size))
            br = tf.get_variable("bias_r", (1, self.config.hidden_size))

            W = tf.get_variable("W", (self.config.embed_size+self.target_vocab_size, self.config.hidden_size))
            U = tf.get_variable("U", (self.config.hidden_size, self.config.hidden_size))
            bh = tf.get_variable("bias_h", (1, self.config.hidden_size))

        with tf.variable_scope("memory/answer", initializer=_xavier_weight_init()):

            U_p = tf.get_variable("U", (self.config.embed_size, self.target_vocab_size))
            b_p = tf.get_variable("bias_p", (self.target_vocab_size,))

    def add_decode_variables(self):

        '''based on github.com/tensorflow/tensorflow/issues/5608#issuecomment-260549420'''
        self.total_input_hops = self.config.num_hops   #version for if you want to set it
        with tf.variable_scope("memory/decode", initializer=_xavier_weight_init()):
            untied_weights = tf.get_variable("W_t", (self.total_input_hops, 2*self.config.hidden_size, self.config.hidden_size))
            untied_biases = tf.get_variable("bias_t", (self.total_input_hops, self.config.hidden_size,))
            #'''

        # The clear_after_read variable must be False, otherwise the TA will 
        # only allow you to read from that index once.
        self.weight_container = tf.TensorArray(tf.float32, self.total_input_hops,
                                     clear_after_read=False, dynamic_size=None, name="w_container")
        
        self.bias_container = tf.TensorArray(tf.float32, self.total_input_hops,
                                     clear_after_read=False, dynamic_size=None, name="b_container")



        # This initialises the TensorArray with the weights broken up in to pieces.
        # The reason this has to be a TensorArray is so that we can index it with a tensor(!)
        self.weight_container = self.weight_container.unpack(untied_weights)
        self.bias_container = self.bias_container.unpack(untied_biases)


    def get_predictions(self, output):
        """Get answer predictions from output"""

        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred
      
    def add_loss_op(self, output):
        """Calculate loss"""
        # optional strong supervision of attention with supporting facts

        gate_loss = 0
        '''
        if self.config.strong_supervision:
            for i, att in enumerate(self.attentions):
                labels = tf.gather(tf.transpose(self.rel_label_placeholder), 0)
                gate_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(att, labels))

        loss = self.config.beta*tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(output, self.answer_placeholder)) + gate_loss
        '''
        logits = tf.transpose(tf.pack(output), perm=[1,0,2])
        loss = sequence_loss_tensor(logits, self.target_placeholder, tf.sign(tf.to_float(self.target_placeholder)), self.target_vocab_size)

        # add l2 regularization for all variables except biases
        for v in tf.trainable_variables():
            if 'bias' not in v.name.lower():
                loss += self.config.l2*tf.nn.l2_loss(v)

        tf.scalar_summary('loss', loss)

        return loss
        
    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        gvs = opt.compute_gradients(loss)

        # optionally cap and noise gradients to regularize
        if self.config.cap_grads:
            gvs = [(tf.clip_by_norm(grad, self.config.max_grad_val), var) for grad, var in gvs]
        if self.config.noisy_grads:
            gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]

        train_op = opt.apply_gradients(gvs)
        return train_op

    def get_input_representation(self, embeddings):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        inputs = tf.nn.embedding_lookup(embeddings, self.input_placeholder)


        #https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L1005
        #https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/kernel_tests/rnn_cell_test.py#L382

        encoded_inputs=[]
        inputs = tf.split(1, self.max_input_len, inputs)
        inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in inputs]

        '''need to change this into tf.while_loop()'''

        '''tf.make_template can solve this sharing issue'''
        shared_bidirectional_dynamic_rnn = tf.make_template('shared_bidirectional_dynamic_rnn', tf.nn.bidirectional_dynamic_rnn)

        for idx, raw_i in enumerate(inputs):
            '''MIGHT WANT USE WAVENET/BYTENET ENCODER HERE INSTEAD BECAUSE IT SEEMS TO HAVE IMPLICIT ALIGNMENT'''
            '''https://arxiv.org/pdf/1601.06733.pdf apparently being able to attend to individual words (LSTMN h_states) (as oppsosed to last hstate) after attentive encoding only improves by 0.2%, so not worth adding intra for after encode, only during'''

            pre_fact_fw_and_bw, _ = shared_bidirectional_dynamic_rnn(self.intra_attention_GRU_cell_fw, self.intra_attention_GRU_cell_bw, raw_i, dtype=np.float32, sequence_length=get_seq_length(raw_i))

            #they reversed it twice, once before and once after: look at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py#L672 and L663
            fw_encode = tf.transpose(pre_fact_fw_and_bw[0], perm=[1,0,2])[-1] #hidden state at last word of sent when conditioned 0 to _length_
            bw_encode = tf.transpose(pre_fact_fw_and_bw[1], perm=[1,0,2])[0]  #hidden state at first word of sent when conditioned _length_ to 0

            encoded_inputs.append(fw_encode + bw_encode)

        pre_facts = tf.transpose(tf.pack(encoded_inputs), perm=[1,0,2])

        '''WOULD PUTTING INTRA SENTENCE ATTN HERE BE REDUNDANT OR PERFORMANT'''
        '''do you need this scope?'''
        outputs_fw_and_bw, _ = tf.nn.bidirectional_dynamic_rnn(self.gru_cell, self.gru_cell, pre_facts, dtype=np.float32, sequence_length=get_seq_length(pre_facts))


        fact_vecs = outputs_fw_and_bw[0] + outputs_fw_and_bw[1]

        fact_vecs = tf.nn.dropout(fact_vecs, self.dropout_placeholder)

        return fact_vecs

    def get_attention(self, prev_memory, fact_vec):
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=True, initializer=_xavier_weight_init()):

            W_1 = tf.get_variable("W_1")
            b_1 = tf.get_variable("bias_1")

            W_2 = tf.get_variable("W_2")
            b_2 = tf.get_variable("bias_2")

            #features = [fact_vec*q_vec, fact_vec*prev_memory, tf.abs(fact_vec - q_vec), tf.abs(fact_vec - prev_memory)]
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

    def normal_GRU_step(self, rnn_input, h):
        """Implement normal GRU"""
        with tf.variable_scope("normal_gru", reuse=True, initializer=_xavier_weight_init()):

            Wu = tf.get_variable("Wu")
            Uu = tf.get_variable("Uu")
            bu = tf.get_variable("bias_u")

            Wr = tf.get_variable("Wr")
            Ur = tf.get_variable("Ur")
            br = tf.get_variable("bias_r")

            W = tf.get_variable("W")
            U = tf.get_variable("U")
            bh = tf.get_variable("bias_h")

            u = tf.sigmoid(tf.matmul(rnn_input, Wu) + tf.matmul(h, Uu) + bu)
            r = tf.sigmoid(tf.matmul(rnn_input, Wr) + tf.matmul(h, Ur) + br)
            h_hat = tf.tanh(tf.matmul(rnn_input, W) + r*tf.matmul(h, U) + bh)
            rnn_output = u*h_hat + (1-u)*h

            return rnn_output

    def generate_episode(self, memory, fact_vecs):
        """Generate episode by applying attention to current fact vectors through a modified GRU"""

        fact_vecs_t = tf.unpack(tf.transpose(fact_vecs, perm=[1,0,2]))

        attentions = [tf.squeeze(self.get_attention(memory, fv), squeeze_dims=[1]) for fv in fact_vecs_t]

        attentions = tf.transpose(tf.pack(attentions))

        self.attentions.append(attentions)

        softs = tf.nn.softmax(attentions)
        softs = tf.split(1, self.max_input_len, softs)
        
        gru_outputs = []

        # set initial state to zero
        h = tf.zeros((self.config.batch_size, self.config.hidden_size))

        # use attention gru
        for i, fv in enumerate(fact_vecs_t):
            h = self._attention_GRU_step(fv, h, softs[i])
            gru_outputs.append(h)

        # extract gru outputs at proper index according to input_lens
        gru_outputs = tf.pack(gru_outputs)
        gru_outputs = tf.transpose(gru_outputs, perm=[1,0,2])

        episode = _last_relevant(gru_outputs, self.input_len_placeholder)

        return episode

    def decoder_step(self, rnn_output):
        """Linear softmax answer module"""
        with tf.variable_scope("answer", reuse=True, initializer=_xavier_weight_init()):

            rnn_output = tf.nn.dropout(rnn_output, self.dropout_placeholder)

            U_p = tf.get_variable("U")
            b_p = tf.get_variable("bias_p")

            vocab_probs = tf.matmul(rnn_output, U_p) + b_p

            output_probs = tf.nn.softmax(vocab_probs)

            return vocab_probs, output_probs

    def attention_decode_for_each_output_step(self, prev_a_all, prev_y_all, fact_vecs):
        '''based on bradbury/metamind concatenate in decoder of their translator http://www.statmt.org/wmt16/pdf/W16-2308.pdf'''
        '''Learning to Translate in Real-time with Neural Machine Translation https://arxiv.org/abs/1610.00388'''
        '''^FIGURE OUT HOW THEIR Y-LSTM PREVENTS MODEL FROM HAVING TO BE INITIALIZED WITH FINAL STATE OF ENCODER'''

        print('==> build episodic memory')

        # generate n_hops episodes
        '''prev_memory = q_vec'''
        prev_memory = prev_a_all[-1]
        prev_token = prev_y_all[-1]

        concat_all = []
        '''maybe add an extra non-linear projection layer like metamind co-attention qa; or maybe don't because it already is using differemt weights than decoder'''
        for i in range(len(prev_a_all)):
            concat_all.append(tf.concat(1, [prev_a_all[i], prev_y_all[i]]))

        '''still need to add gate_mechanism if you want it to be able to forget facts'''
        '''should maybe use make_template for all variables/weights declared in Adaptive_Episodes'''
        ''' add modifications that he mentions in reddit.com/r/MachineLearning/comments/59sfz8/research_learning_to_reason_with_adaptive/d9bgqxw/'''

        episode_module = Adaptive_Episodes(Adaptive_Episodes_Config())

        last_mem, remainders, iterations = episode_module.do_generate_episodes(prev_memory, fact_vecs, \
            self.config.batch_size, self.config.hidden_size, self.max_input_len, self.input_len_placeholder, \
            self.total_input_hops, self.config.epsilon, self.weight_container, self.bias_container)

        iters = tf.Print(iterations, [iterations], message="Iterations: ", summarize=20)
        rem = tf.Print(remainders, [remainders], message="remainders: ", summarize=20)
        lm = tf.Print(last_mem, [last_mem], message="last_mem: ", summarize=20)

        concat = concat_all[-1]

        h_state = self.normal_GRU_step(concat, last_mem)
        vocab_probs, output_probs = self.decoder_step(h_state)

        return h_state, output_probs, vocab_probs, iterations, remainders

    def inference(self):
        """Performs inference on the DMN model"""

        # set up embedding
        embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")
        '''^i think this is based on input vocab size'''
        
        with tf.variable_scope("input", initializer=_xavier_weight_init()):
            print('==> get input representation')
            fact_vecs = self.get_input_representation(embeddings)

        # keep track of attentions for possible strong supervision
        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=_xavier_weight_init(), reuse=None):
            '''you have an arbitrary length and prev_a and prev_y here for now'''

            prev_a = tf.transpose(fact_vecs, perm=[1,0,2])[-1]
            prev_y = tf.zeros([self.config.batch_size, self.target_vocab_size])

            output=[]
            prev_a_all=[]
            prev_y_all=[]
            arbitrary_num = 2

            for i in range(0, self.max_t_len):
                prev_a_all.append(prev_a)
                prev_y_all.append(prev_y)
                prev_a, prev_y, vocab_probs, attn_iters_step, attn_halt_probs_step = self.attention_decode_for_each_output_step(prev_a_all, prev_y_all, fact_vecs)
                output.append(vocab_probs)
                self.attn_iters.append(attn_iters_step)
                self.attn_halt_probs.append(attn_halt_probs_step)

        return output

    def target_id_to_vocab_w_new_line(self, id_num):
        if id_num == 8:
            return '\n'
        else:
            return self.target_id_to_vocab[id_num]

    def execute_prediction(self, code):
        exec(code)

        return exec_output


    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):

        '''NEED TO ADD BATCH_NORM OR LAYER NORM'''

        config = self.config
        dp = config.dropout
        bn = config.batch_norm
        ln = config.layer_norm
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
            bn = 1

        total_steps = len(data[0]) / config.batch_size
        total_loss = []
        accuracy = 0
        
        # shuffle data
        p = np.random.permutation(len(data[0]))
        tp, ip, tl, il, im = data
        tp, ip, tl, il, im = tp[p], ip[p], tl[p], il[p], im[p] 

        print(total_steps)
        print(list(range(total_steps)))

        for step in range(total_steps):
            index = list(range(step*config.batch_size,(step+1)*config.batch_size))

            feed = {self.target_placeholder: tp[index],
                  self.input_placeholder: ip[index],
                  self.target_len_placeholder: tl[index],
                  self.input_len_placeholder: il[index],
                  self.dropout_placeholder: dp}
            loss, pred, summary, _, attn_iters_out, attn_halt_probs_out = session.run(
              [self.calculate_loss, self.pred_seq, self.merged, train_op, self.attn_iters, self.attn_halt_probs], feed_dict=feed)

            if train_writer is not None:
                train_writer.add_summary(summary, num_epoch*total_steps + step)


            '''MINIPY CAUSES SOME ANSWER TO BECOME DUPLICATES; NEED TO REMOVE DUPLICATES'''

            targets = tp[step*config.batch_size:(step+1)*config.batch_size]

            accuracy += np.sum(pred == targets)/float(len(targets))

            total_loss.append(loss)

            if verbose and step % verbose == 0:
                sys.stdout.write('\n{} / {} : loss = {}\n\n'.format(
                  step, total_steps, np.mean(total_loss)))
                #sys.stdout.flush()

        if verbose:
            #sys.stdout.write('\r')
            sys.stdout.write('\n')

        return np.mean(total_loss), accuracy/float(total_steps)


    def __init__(self, config):

        self.config = config
        self.attn_iters = []
        self.attn_halt_probs = []
        self.variables_to_save = {}
        self.load_data(debug=False)
        self.add_placeholders()
        self.add_reused_variables()
        self.add_decode_variables()
        self.output = self.inference()
        self.pred = self.get_predictions(self.output)

        self.pred_seq = [self.get_predictions(i) for i in self.output]

        self.calculate_loss = self.add_loss_op(self.output)
        self.train_step = self.add_training_op(self.calculate_loss)
        self.merged = tf.merge_all_summaries()

