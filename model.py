import tensorflow as tf
import numpy as np
import os
import cPickle as pickle
from flags import Flags
import preprocess
import sys
import logging
from tensorflow.contrib.session_bundle import exporter

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Params():
    max_seq_len = 33
    emb_dim = 100
    topic_dim = 10
    hidden_size = 100
    batch_size = 1
    valid_batch_size = 1
    test_batch_size = valid_batch_size
    max_epochs = {Flags.TRAIN: 200, Flags.TEST: 1}
    model_path = 'models/'
    model_name = os.path.join(model_path, 'rnn_from_annotation_framework')
    max_grad_norm = 10
    init_scale = 0.1
    train_batches = None
    valid_batches = None
    test_batches = None


class Stats():
    vocab_size = None
    file_len_count = None
    num_classes = None


data = {}


def pad_to_max_len(con, yy, batch_size):
    lens_con = []
    arr_con = np.zeros((batch_size, Params.max_seq_len))
    ys = np.zeros((batch_size, Stats.num_classes))

    for i, (seq_con, y) in enumerate(zip(con, yy)):
        lens_con.append(len(seq_con))
        arr_con[i, :len(seq_con)] = seq_con
    ys[np.arange(batch_size), yy] = 1
    return arr_con, ys, np.array(lens_con)


class RNNclassifier():
    def __init__(self, flag_is_train):
        (kp, batch_size) = (0.6, Params.batch_size) if flag_is_train == Flags.TRAIN else (1.0, Params.valid_batch_size)
        self.context = tf.placeholder(dtype=tf.int32, shape=[None, Params.max_seq_len], name="CONTEXT")
        self.flag_one_hot = tf.placeholder(tf.float32, shape=[None, Stats.num_classes], name="FLAG")
        self.sequence_length_context = tf.placeholder(tf.int32, shape=[None], name="SEQ_LEN_CONTEXT")

        self.flag_int = tf.cast(tf.arg_max(self.flag_one_hot, dimension=1, name='GET_RESPONSE_ID'), tf.int32)
        self.embedding_matrix = tf.get_variable("EMBEDDING_MATRIX", shape=[Stats.vocab_size, Params.emb_dim],
                                                dtype=tf.float32,
                                                initializer=tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32),
                                                trainable=False)
        self.context_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, self.context, name="CONTEXT_EMB")

        self.lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.LSTMCell(Params.hidden_size, input_size=Params.emb_dim), kp, kp)
        self.lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.LSTMCell(Params.hidden_size, input_size=Params.emb_dim), kp, kp)

        self.init_hidden = self.lstm_cell_fw.zero_state(batch_size, dtype=tf.float32)
        _, context_states = tf.nn.bidirectional_dynamic_rnn(self.lstm_cell_fw, self.lstm_cell_bw,
                                                            self.context_embeddings,
                                                            sequence_length=self.sequence_length_context,
                                                            initial_state_fw=self.init_hidden,
                                                            initial_state_bw=self.init_hidden, scope="CONTEXT")

        context_states = context_states[0].h + context_states[1].h

        M1 = tf.get_variable("M", shape=[Params.hidden_size, Stats.num_classes * 2], dtype=tf.float32,
                             initializer=tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32))
        b1 = tf.get_variable("b", shape=[Stats.num_classes * 2], dtype=tf.float32,
                             initializer=tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32))

        M2 = tf.get_variable("M2", shape=[Stats.num_classes * 2, Stats.num_classes], dtype=tf.float32,
                             initializer=tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32))
        b2 = tf.get_variable("b2", shape=[Stats.num_classes], dtype=tf.float32,
                             initializer=tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32))

        predictions = tf.matmul(tf.nn.relu(tf.matmul(context_states, M1) + b1), M2) + b2

        self.sig_predictions = tf.nn.softmax(predictions)
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(predictions, self.flag_one_hot, name="LOSS"))
        self.matches_topk = tf.nn.in_top_k(predictions, self.flag_int, 1, name='TOP_K')
        self.pred_top1 = tf.cast(tf.argmax(self.sig_predictions, axis=1), tf.int32)
        self.matches = tf.equal(self.flag_int, self.pred_top1)
        self.accuracy = tf.reduce_mean(tf.cast(self.matches_topk, tf.float32))
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), Params.max_grad_norm)
        optimizer = tf.train.AdamOptimizer()

        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        return


def run_epoch(flag, sess, model_train, utterances, targets, valid_utterances, valid_targets, model_saver, max_acc):

    epoch_cost = 0
    total_acc = 0

    for batch_i in range(Params.train_batches):
        start = batch_i * Params.batch_size
        end = start + Params.batch_size
        batch_utterances = utterances[start:end]
        batch_targets = targets[start:end]
        arr_utterances, arr_targets, seq_lens = pad_to_max_len(batch_utterances, batch_targets,
                                                               batch_size=Params.batch_size)
        _, loss, acc, sig_pred, matches, matches_topk = sess.run(
            [model_train.train_op, model_train.loss, model_train.accuracy, model_train.sig_predictions,
             model_train.matches, model_train.matches_topk],
            {model_train.context: arr_utterances, model_train.flag_one_hot: arr_targets,
             model_train.sequence_length_context: seq_lens})
        # print(acc)
        total_acc += acc
        epoch_cost += loss
    valid_acc = run_test_epoch(flag, sess, model_train, valid_utterances, valid_targets)
    if (valid_acc > max_acc):
        max_acc = valid_acc
        print("max_acc: ", max_acc)
        print("valid_acc: ", valid_acc)
        print("SAVING MODEL")
        model_saver.save(sess, save_path=Params.model_name, latest_filename='checkpoints')
    print("accuracy: ", total_acc / float(Params.train_batches))
    print("loss: ", epoch_cost)
    return max_acc


def run_test_epoch(flag, sess, model_train, utterances, targets):
    epoch_cost = 0
    total_acc = 0
    if (flag == Flags.TRAIN):
        num_batches = Params.valid_batches
    else:
        num_batches = Params.test_batches
    for batch_i in range(num_batches):
        start = batch_i * Params.valid_batch_size
        end = start + Params.valid_batch_size
        batch_utterances = utterances[start:end]
        batch_targets = targets[start:end]
        arr_utterances, arr_targets, seq_lens = pad_to_max_len(batch_utterances, batch_targets, Params.valid_batch_size)
        loss, acc = sess.run([model_train.loss, model_train.accuracy],
                             {model_train.context: arr_utterances, model_train.flag_one_hot: arr_targets,
                              model_train.sequence_length_context: seq_lens})
        print(acc)
        total_acc += acc
        epoch_cost += loss
    print('valid ' if flag == Flags.TRAIN else 'test' + " acc: {}".format(total_acc / float(num_batches)))
    return total_acc


def export_model(sess, model_train):
    init_op = tf.group(tf.initialize_all_tables(), name='init_op')
    saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(saver)

    model_exporter.init(sess.graph.as_graph_def(), init_op=init_op, named_graph_signatures={
        'inputs': exporter.generic_signature(
            {'batch_utterances': model_train.context,
             'batch_target': model_train.flag_one_hot,
             'batch_seq_lens': model_train.sequence_length_context}),
        'outputs': exporter.generic_signature({'loss': model_train.loss})})
    model_exporter.export('exported_model/', tf.constant(10), sess)
    return


def run_model(flag, load_from_existing):
    valid_utterances, valid_targets, utterances, targets, vocab, response_vocab, emb_matrix = preprocess.preprocess_data(
        flag, load_from_existing)
    Stats.vocab_size = len(vocab)
    Stats.num_classes = len(response_vocab)
    Params.train_batches = len(utterances) // Params.batch_size
    Params.valid_batches = len(valid_utterances) // Params.valid_batch_size

    if (flag == Flags.TEST):
        Params.test_batches = len(utterances) // Params.test_batch_size
        Params.train_batches = None

    initializer = tf.random_uniform_initializer(-Params.init_scale, Params.init_scale)
    max_acc = 0.
    sess = tf.Session()

    for i in range(1):
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model_train = RNNclassifier(flag)
        model_saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(Params.model_path, latest_filename='checkpoints')
        if ckpt and ckpt.model_checkpoint_path:
            logger.info('LOADING SAVED MODEL')
            print("Loading model from :{}".format(ckpt.model_checkpoint_path))
            model_saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            if (flag == Flags.TEST):
                print "NO SAVED MODEL TO TEST; EXITING"
                sys.exit(1)
            with sess.as_default():
                tf.initialize_all_variables().run()
            logger.info('CREATED MODEL WITH FRESH PARAMETERS')
            sess.run(tf.assign(model_train.embedding_matrix, emb_matrix))
            # export_model(sess, model_train)

    for k in range(Params.max_epochs[flag]):
        print("EPOCH NUMBER {}".format(k + 1))
        if (flag == Flags.TRAIN):
            max_acc = run_epoch(flag, sess, model_train, utterances, targets, valid_utterances, valid_targets,
                                model_saver, max_acc)
        else:
            max_acc = run_test_epoch(flag, sess, model_train, utterances, targets)
    return


def run():
    '''try:
        flag = Flags.TRAIN if sys.argv[1]=='train' else Flags.TEST
        reuse = True if sys.argv[2] == 'true' else False
    except IndexError:
        print('CORRECT USAGE: python run.py <train_flag> <reuse_flag>')
        sys.exit(1)
    '''
    os.chdir('../')
    run_model(Flags.TRAIN, False)


if __name__ == '__main__':
    run()
