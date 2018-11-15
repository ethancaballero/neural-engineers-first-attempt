import sys
import time

import pdb
import numpy as np
from copy import deepcopy

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

from tensorflow.contrib.layers import fully_connected

from rnn_decoder_utils import rnn_decoder_simple
import data_utils

class Config(object):
    """Holds model hyperparams and data information."""

    batch_size = 25

    embed_size = 80
    hidden_size = 80

    max_epochs = 256
    early_stopping = 20

    dropout = 0.9
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

    num_hops = 3
    num_attention_features = 2

    max_allowed_inputs = 130
    #num_train = 9000
    num_train = 78

    floatX = np.float32

    babi_id = "1"
    babi_test_id = ""

    train_mode = True

def get_seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

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
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
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

        #"""
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
        """Adds trainable variables which are later (not?) reused"""  
        for i in range(self.config.num_hops): 
            with tf.variable_scope("memory/decode" + "/" + str(i), initializer=_xavier_weight_init()):
                Wt = tf.get_variable("W_t", (2*self.config.hidden_size, self.config.hidden_size))
                bt = tf.get_variable("bias_t", (self.config.hidden_size,))


    def get_predictions(self, output):
        """Get answer predictions from output"""

        """pred is going to have to have number added to it depending on which rnn weight space is used for selection"""
        """or self.answer_placeholder is going to have number sutracted from it depending on which rnn weight space is used for selection"""

        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred
      
    def add_loss_op(self, output):
        """Calculate loss"""

        gate_loss = 0
        '''
        if self.config.strong_supervision:
            for i, att in enumerate(self.attentions):
                labels = tf.gather(tf.transpose(self.rel_label_placeholder), 0)
                gate_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(att, labels))
                '''

        '''
        loss = self.config.beta*tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(output, self.answer_placeholder)) + gate_loss
        '''

        loss = 0

        # (B, decode_length) -> (decode_length, B)
        m_test = tf.transpose(self.target_placeholder, perm=[1,0])

        for idx, single_output in enumerate(output):
            loss += self.config.beta * tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(output[idx], m_test[idx])) + gate_loss

        # add l2 regularization for all variables except biases
        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
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

        '''EXPERIMENT WITH PUTTING INTRA SENTENCE ATTN ON bi-directional WRAPPER AS OPPOSED TO GRU WRAPPER'''
        '''and maybe allow reverse phase during birnn to attend to forward phases h_states'''

        print self.intra_attention_GRU_cell_fw
        encoded_inputs=[]
        inputs = tf.split(1, self.max_input_len, inputs)
        inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in inputs]

        '''need to change this into tf.while_loop()'''

        '''tf.make_template can solve this sharing issue'''
        shared_bidirectional_dynamic_rnn = tf.make_template('shared_bidirectional_dynamic_rnn', tf.nn.bidirectional_dynamic_rnn)

        for idx, raw_i in enumerate(inputs):
            '''MIGHT WANT USE WAVENET/BYTENET ENCODER HERE INSTEAD BECAUSE IT SEEMS TO HAVE IMPLICIT ALIGNMENT'''
            '''https://arxiv.org/pdf/1601.06733.pdf apparently being able to attend to individual words (LSTMN h_states) (as oppsosed to last hstate) after attentive encoding only improves by 0.2%, so not worth adding intra for after encode, only during'''

            pre_fact_fw_and_bw, _ = shared_bidirectional_dynamic_rnn(self.intra_attention_GRU_cell_fw, self.intra_attention_GRU_cell_fw, raw_i, dtype=np.float32, sequence_length=get_seq_length(raw_i))

            #they reversed it twice, once before and once after: look at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py#L672 and L663
            fw_encode = tf.transpose(pre_fact_fw_and_bw[0], perm=[1,0,2])[-1] #hidden state at last word of sent when conditioned 0 to _length_
            bw_encode = tf.transpose(pre_fact_fw_and_bw[1], perm=[1,0,2])[0]  #hidden state at first word of sent when conditioned _length_ to 0

            encoded_inputs.append(fw_encode + bw_encode)

        pre_facts = tf.transpose(tf.pack(encoded_inputs), perm=[1,0,2])

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

        print '==> build episodic memory'
        # generate n_hops episodes
        '''prev_memory = q_vec'''
        prev_memory = prev_a_all[-1]
        prev_token = prev_y_all[-1]

        concat_all = []
        '''maybe add an extra non-linear projection layer like metamind co-attention qa; or maybe don't because it already is using differemt weights than decoder'''
        for i in range(len(prev_a_all)):
            concat_all.append(tf.concat(1, [prev_a_all[i], prev_y_all[i]]))

        '''amount of hops here will be decided by threshold/reinforcement?'''
        for i in range(self.config.num_hops):
            # get a new episode
            print '==> generating episode', i
            episode = self.generate_episode(prev_memory, fact_vecs)

            ''' # target_side attention
            episode = self.generate_episode(prev_memory, fact_vecs, concat_all)
            '''

            '''replace this with tf.make_template'''
            with tf.variable_scope("decode" + "/" + str(i), initializer=_xavier_weight_init(), reuse=True):
                Wt = tf.get_variable("W_t")
                bt = tf.get_variable("bias_t")

            prev_memory = tf.nn.relu(tf.matmul(tf.concat(1, [prev_memory, episode]), Wt) + bt)

        last_mem = prev_memory

        concat = concat_all[-1]

        h_state = self.normal_GRU_step(concat, last_mem)
        vocab_probs, output_probs = self.decoder_step(h_state)

        return h_state, output_probs, vocab_probs

    def inference(self):
        """Performs inference on the DMN model"""

        # set up embedding
        embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")
        '''^i think this is based on input vocab size'''
        
        with tf.variable_scope("input", initializer=_xavier_weight_init()):
            print '==> get input representation'
            fact_vecs = self.get_input_representation(embeddings)

        # keep track of attentions for possible strong supervision
        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=_xavier_weight_init(), reuse=None):
            '''you have an arvitrary length and prev_a and prev_y here for now'''

            prev_a = tf.transpose(fact_vecs, perm=[1,0,2])[-1]
            prev_y = tf.zeros([self.config.batch_size, self.target_vocab_size])

            output=[]
            prev_a_all=[]
            prev_y_all=[]
            arbitrary_num = 5
            for i in range(0, arbitrary_num):
                print i
                prev_a_all.append(prev_a)
                prev_y_all.append(prev_y)
                prev_a, prev_y, vocab_probs = self.attention_decode_for_each_output_step(prev_a_all, prev_y_all, fact_vecs)
                output.append(vocab_probs)

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
        if train_op is None:
            train_op = tf.no_op()
            dp = 1

        total_steps = len(data[0]) / config.batch_size
        total_loss = []
        accuracy = 0
        
        # shuffle data
        p = np.random.permutation(len(data[0]))
        tp, ip, tl, il, im = data
        tp, ip, tl, il, im = tp[p], ip[p], tl[p], il[p], im[p] 

        for step in range(total_steps):
            index = range(step*config.batch_size,(step+1)*config.batch_size)

            feed = {self.target_placeholder: tp[index],
                  self.input_placeholder: ip[index],
                  self.target_len_placeholder: tl[index],
                  self.input_len_placeholder: il[index],
                  self.dropout_placeholder: dp}
            loss, pred, summary, _ = session.run(
              [self.calculate_loss, self.pred_seq, self.merged, train_op], feed_dict=feed)

            if train_writer is not None:
                train_writer.add_summary(summary, num_epoch*total_steps + step)

            '''IS ACCURACY RIGHT, DOES IT STILL WORK NOW THAT YOU'VE SWITCHED FROM TOKEN TO SEQUENCE''' 
            targets = tp[step*config.batch_size:(step+1)*config.batch_size]


            accuracy += np.sum(pred == targets)/float(len(targets))

            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\n{} / {} : loss = {}\n\n'.format(
                  step, total_steps, np.mean(total_loss)))
                #sys.stdout.flush()


        if verbose:
            sys.stdout.write('\n')

        return np.mean(total_loss), accuracy/float(total_steps)


    def __init__(self, config):

        self.config = config
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

