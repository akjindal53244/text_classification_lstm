import tensorflow as tf
import numpy as np
import os
import sys

from tensorflow.contrib.session_bundle import exporter

import pre_train_word
import pickle
from datetime import datetime

init_scale = -0.01

class Stats():
    word_vocab_size= 10000
    pgm_vocab_size = 10000
    num_classes = 100

class Params():
    emb_dim = 300
    batch_size = 5
    hidden_dim = 512
    max_grad_norm = 5
    max_seq_length = 400
    train_batches = None
    num_epochs = {'train' :50,'test':1}
    num_batches = None

class DoubleRNN():
    def __init__(self,kp=1.0):
        self.a1 = tf.placeholder(dtype=tf.int32, shape=[None, Params.max_seq_length], name = "CONTEXT")
        self.u1= tf.placeholder(dtype=tf.int32, shape=[None, Params.max_seq_length], name = "CONTEXT_PGM")
        self.a2= tf.placeholder(dtype=tf.int32, shape=[None, Params.max_seq_length], name = "CONTEXT2")
        self.u2= tf.placeholder(dtype=tf.int32, shape=[None, Params.max_seq_length], name="CONTEXT2_PGM")
        self.a1_seq_length = tf.placeholder(dtype=tf.int32, shape=[None], name="A1_SEQ_LEN")
        self.u1_seq_length = tf.placeholder(dtype=tf.int32, shape=[None], name="U1_SEQ_LEN")
        self.a2_seq_length = tf.placeholder(dtype=tf.int32, shape=[None], name="A2_SEQ_LEN")
        self.u2_seq_length = tf.placeholder(dtype=tf.int32, shape=[None], name="U2_SEQ_LEN")

        self.response = tf.placeholder(dtype=tf.int32, shape=[None], name = "RESPONSE")
        self.num_contexts = tf.placeholder(dtype=tf.int32, shape=[None], name="NUM_CONTEXTS")

        self.target = tf.one_hot(self.response, Stats.num_classes)
        # self.embedding_matrix = tf.get_variable("EMBEDDING_MATRIX", shape=[Stats.vocab_size, Params.word_emb_dim],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32), trainable=False)
        self.embedding_matrix = tf.get_variable("EMBEDDING_MATRIX", shape=[Stats.word_vocab_size, Params.emb_dim],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32), trainable=False)

        a1_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, self.a1, name="A1")
        u1_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, self.u1, name="U1")
        a2_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, self.a2, name="A2")
        u2_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, self.u2, name="U2")






        self.a_lstm_cell = tf.nn.rnn_cell.LSTMCell(Params.hidden_dim,input_size=Params.emb_dim ,initializer=tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32))
        self.u_lstm_cell = tf.nn.rnn_cell.LSTMCell(Params.hidden_dim,input_size=Params.emb_dim ,initializer=tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32))
        self.init_hidden = self.a_lstm_cell.zero_state( Params.batch_size, dtype=tf.float32)

        with tf.variable_scope("A_LSTM") as vs:
            _, a1_states =tf.nn.dynamic_rnn(self.a_lstm_cell, inputs = a1_embeddings, sequence_length=self.a1_seq_length, initial_state=self.init_hidden)
            vs.reuse_variables()
            _, a2_states =tf.nn.dynamic_rnn(self.a_lstm_cell, inputs = a2_embeddings, sequence_length=self.a2_seq_length, initial_state=self.init_hidden)
        with tf.variable_scope("U_LSTM") as vs:
            _, u1_states =tf.nn.dynamic_rnn(self.a_lstm_cell, inputs = u1_embeddings, sequence_length=self.u1_seq_length, initial_state=self.init_hidden)
            vs.reuse_variables()
            _, u2_states =tf.nn.dynamic_rnn(self.a_lstm_cell, inputs = u2_embeddings, sequence_length=self.u2_seq_length, initial_state=self.init_hidden)


        self.utterance_level_lstm_cell = tf.nn.rnn_cell.LSTMCell(1024,input_size=Params.hidden_dim ,initializer=tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32), num_proj=Stats.num_classes)
        _, utterance_level_states = tf.nn.dynamic_rnn(self.utterance_level_lstm_cell, inputs = tf.pack([a1_states.h, u1_states.h, a2_states.h, u2_states.h], axis=1), sequence_length=self.num_contexts, initial_state=self.utterance_level_lstm_cell.zero_state(Params.batch_size, dtype=tf.float32))
        final_outputs = utterance_level_states.h
        self.predictions = final_outputs

        self.repeated_predictions = tf.concat(0, [self.predictions for _ in range(5)])
        self.cost_per_5_arr = tf.nn.softmax_cross_entropy_with_logits(self.repeated_predictions, self.target, name="LOSS_PER_5")

        top_k = tf.nn.in_top_k(self.predictions, self.response, 1, name="IN_TOP_K")
        top_k_float = tf.cast(top_k, tf.float32, name="TOP_K_FLOAT")
        self.recall = tf.reduce_sum(top_k_float)
        # self.neg_predictions = tf.neg(self.predictions)
        self.argmaxes = tf.arg_max(self.predictions, dimension=1, name="ARGMAXES")

        self.matches = tf.reduce_sum(tf.cast(tf.equal(self.argmaxes, tf.cast(self.response, tf.int64)),tf.float32))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predictions, self.target, name= "LOSS"))
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), Params.max_grad_norm)
        optimizer = tf.train.AdamOptimizer()

        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        return

def pad_to_max_len(con, res,yy):
    lens_con = []
    lens_res = []
    arr_con = np.zeros((Params.batch_size, Params.max_seq_length))
    arr_res = np.zeros((Params.batch_size, Params.max_seq_length))
    ys = np.zeros((Params.batch_size))
    for i, (seq_con, seq_res,y) in enumerate(zip(con, res, yy)):
        seq_con_len = len(seq_con)
        seq_res_len = len(seq_res)
        lens_con.append(len(seq_con))
        lens_res.append(len(seq_res))
        arr_con[i,:min(seq_con_len, Params.max_seq_length)] =  seq_con[:min(seq_con_len, Params.max_seq_length)]
        arr_res[i,:min(seq_res_len, Params.max_seq_length)] = seq_res[:min(seq_res_len, Params.max_seq_length)]
        ys[i] = y
    return arr_con, arr_res, ys, np.array(lens_con), np.array(lens_res)


def run_epoch(flag,data, sess, model_train, min_cost, model_saver):
    epoch_cost = 0
    total_acc = 0
    total_recall =0
    if(flag == 'train'):
        for batch_i in range(Params.num_batches):
            start = batch_i * Params.batch_size
            end = start + Params.batch_size

            a1 = data['a1'][start:end]
            u1 = data['u1'][start:end]
            a2 = data['a2'][start:end]
            u2 = data['u2'][start:end]
            y = data['r'][start:end]
            num_contexts = data['num_contexts'][start:end]

            arr_a1, arr_a2, arr_y, lens_arr_a1, lens_arr_a2 = pad_to_max_len(a1, a2, y)
            arr_u1, arr_u2, arr_num_contexts, lens_arr_u1, lens_arr_u2 = pad_to_max_len(u1,u2,num_contexts)

            cost,_ = sess.run([model_train.loss, model_train.train_op], {model_train.a1 :arr_a1, model_train.a2 :arr_a2, model_train.u1 :arr_u1, model_train.u2:arr_u2, model_train.a1_seq_length: lens_arr_a1, model_train.a2_seq_length: lens_arr_a2, model_train.u1_seq_length: lens_arr_u1, model_train.u2_seq_length: lens_arr_u2, model_train.response : arr_y, model_train.num_contexts : arr_num_contexts})
            epoch_cost += cost
            # print("cost" , cost)
        print("EPOCH_COST",epoch_cost)
        if(epoch_cost < min_cost):
            min_cost = epoch_cost
            model_saver.save(sess, save_path=pre_train_word.Paths.model_name, latest_filename='checkpoints')
    else:
        f= open('output/test_cost.txt','w')
        for batch_i in range(Params.num_batches):
            print(batch_i)
            start = batch_i * Params.batch_size
            end = start + Params.batch_size

            a1 = data['a1'][start:end]
            u1 = data['u1'][start:end]
            a2 = data['a2'][start:end]
            u2 = data['u2'][start:end]
            y = data['r'][start:end]
            num_contexts = data['num_contexts'][start:end]


            arr_a1, arr_a2, arr_y, lens_arr_a1, lens_arr_a2 = pad_to_max_len(a1, a2, y)
            arr_u1, arr_u2, arr_num_contexts, lens_arr_u1, lens_arr_u2 = pad_to_max_len(u1,u2,num_contexts)

            # print("a1",arr_a1)
            # print("a1_len", lens_arr_a1)
            # print("u1", arr_u1)
            # print("u1_len", lens_arr_u1)
            # print("a2", arr_a2)
            # print("a2_len", lens_arr_a2)
            # print("u2", arr_u2)
            # print("u2_len", lens_arr_u2)
            # print("r", arr_y)
            # print("num_contexts", arr_num_contexts)
            start_time = datetime.now()
            cost,recall = sess.run([model_train.loss, model_train.recall], {model_train.a1 :arr_a1, model_train.a2 :arr_a2, model_train.u1 :arr_u1, model_train.u2:arr_u2, model_train.a1_seq_length: lens_arr_a1, model_train.a2_seq_length: lens_arr_a2, model_train.u1_seq_length: lens_arr_u1, model_train.u2_seq_length: lens_arr_u2, model_train.response : arr_y, model_train.num_contexts : arr_num_contexts})
            end_time = datetime.now()
            delta = end_time - start_time
        #    print("time in ms",delta.total_seconds()*1000)
            epoch_cost += cost
            total_recall += recall
            f.write(str(cost) + '\n')
        print('COST '+str(epoch_cost))
        print(float(total_recall)/Params.num_batches)
        f.close()
    return min_cost

def get_accuracy():
    # esc = pickle.load(open('esc_indices.pkl'))
    try:
        corr = pickle.load(open('correct_indices_unique'))
        print(len(corr))
        f=open('correct_losses_syntrain_nattest_unique')
        lines = f.readlines()
        f.close()
        print(len(lines))
        score = 0
        num_ex = len(corr)
        pos_score = 0
        esc_score = 0
        res = [0 for _ in range(5)]
        for i,ind in enumerate(corr):
            res[0] = float(lines[5 * i].rstrip())
            res[1] = float(lines[5*i + 1].rstrip())
            res[2] = float(lines[5*i + 2].rstrip())
            res[3] = float(lines[5*i + 3].rstrip())
            res[4] = float(lines[5*i + 4].rstrip())
            res = np.array(res)
            ind_min = np.argmin(res)
            # if((corr[i] == 0) and res_0 <= res_1):
            #     score+=1
            # elif((corr[i] == 1) and res_0 >= res_1):
            #     score += 1
            if(corr[i] == ind_min):
                score +=1
                # if(i in esc):
                #     esc_score += 1
                # else:
                #     pos_score += 1
        print(float(score)/float(num_ex))
        # print(float(esc_score)/float(len(esc)))
        # print(float(pos_score) / float(num_ex - len(esc)) )

    except IndexError as e:
        print(i)
        print(e.message)
        # print("IndexError")
    # print(num_ex)


def run_model(flag):
    if(flag == 'test'):
        Params.batch_size = 1
        data, dictionaries = pre_train_word.preprocess(flag)
    else:
        data, dictionaries = pre_train_word.preprocess(flag)
    vocab, response_vocab, word_embeddings = dictionaries
    Stats.word_vocab_size = len(vocab)
    Stats.num_classes = len(response_vocab)
    num_ex = len(data['a1'])
    Params.num_batches = num_ex // Params.batch_size

    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    min_cost = sys.float_info.max
    with tf.Session() as sess:

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model_train = DoubleRNN()
        model_saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(pre_train_word.Paths.model_name, latest_filename='checkpoints')
        if ckpt and ckpt.model_checkpoint_path:
            print("Loading model from :{}".format(ckpt.model_checkpoint_path))
            model_saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.initialize_all_variables().run()
            print("CREATED MODEL WITH FRESH PARAMETERS")
        for k in range(Params.num_epochs[flag]):
            min_cost = run_epoch(flag, data,sess, model_train, min_cost, model_saver)

def test_epoch(data, sess, model_train, num_batches =5):
    f = open('output/test_cost.txt','w')
    for batch_i in range(num_batches):
        start = batch_i
        end = start + 1

        a1 = data['a1'][start:end]
        u1 = data['u1'][start:end]
        a2 = data['a2'][start:end]
        u2 = data['u2'][start:end]
        y = data['r'][start:end]
        num_contexts = data['num_contexts'][start:end]

        arr_a1, arr_a2, arr_y, lens_arr_a1, lens_arr_a2 = pad_to_max_len(a1, a2, y)
        arr_u1, arr_u2, arr_num_contexts, lens_arr_u1, lens_arr_u2 = pad_to_max_len(u1, u2, num_contexts)

        cost, recall = sess.run([model_train.loss, model_train.recall],
                                {model_train.a1: arr_a1, model_train.a2: arr_a2, model_train.u1: arr_u1,
                                 model_train.u2: arr_u2, model_train.a1_seq_length: lens_arr_a1,
                                 model_train.a2_seq_length: lens_arr_a2, model_train.u1_seq_length: lens_arr_u1,
                                 model_train.u2_seq_length: lens_arr_u2, model_train.response: arr_y,
                                 model_train.num_contexts: arr_num_contexts})
        f.write(str(cost)+'\n')
    f.close()

    return

def init_test():
    Params.batch_size = 1
    dictionaries = pre_train_word.load_vocab_pkls()
    vocab, response_vocab = dictionaries
    Stats.word_vocab_size = len(vocab)
    Stats.num_classes = len(response_vocab)
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    sess = tf.Session()
    try:
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model_train = DoubleRNN()
        model_saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(pre_train_word.Paths.model_name, latest_filename='checkpoints')
        if ckpt and ckpt.model_checkpoint_path:
            print("Loading model from :{}".format(ckpt.model_checkpoint_path))
            model_saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('NO SAVED MODEL FOUND. RETURNING NONE')
            return None

        ####Tensor-flow serving block starts##############
        init_op = tf.group(tf.initialize_all_tables(), name='init_op')
        saver = tf.train.Saver(sharded=True)

        model_exporter = exporter.Exporter(saver)
        model_exporter.init(
            sess.graph.as_graph_def(),
            init_op=init_op,
            named_graph_signatures={
                'inputs': exporter.generic_signature(
                    {'a1_utter': model_train.a1, 'a2_utter': model_train.a2,
                     'u1_utter': model_train.u1, 'u2_utter': model_train.u2, 'a1_len': model_train.a1_seq_length,
                     'a2_len': model_train.a2_seq_length, 'u1_len': model_train.u1_seq_length,
                     'u2_len': model_train.u2_seq_length, 'response': model_train.response, 'num_contexts':model_train.num_contexts}),
                'outputs': exporter.generic_signature({'loss': model_train.cost_per_5_arr})
            }
        )
        model_exporter.export('/home/phegde/Desktop/amelia-v3-temp/v3-launch-demo-models/hlstm_model', tf.constant(9), sess)
        print("exporting done")
        ####Tensor-flow serving block ends##############

        return sess, model_train, vocab, response_vocab
    except IOError as e:
        print(e)
        return None

init_test()
#run_model('test')
# get_accuracy()
