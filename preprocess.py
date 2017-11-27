import os
import numpy as np
import cPickle as pickle
from collections import Counter
import logging
from flags import Flags

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
handler = logging.FileHandler('vanguard.log')
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


class Paths():
    data_path = 'data'
    glove_path = 'data/glove.6B.100d.pkl'
    train_file = 'train_data.txt'
    valid_file = 'valid_data.txt'
    test_file = 'test.txt'
    train_test_valid_file = 'train2.txt'

    pkl_path = 'dump'
    vocab_filename = 'vocab.pkl'
    response_vocab_filename = 'response_vocab.pkl'
    emb_matrix_filename = 'emb_matrix.pkl'

    train_path = os.path.join(data_path, train_file)
    valid_path = os.path.join(data_path, valid_file)
    test_path = os.path.join(data_path, test_file)
    train_test_valid_path = os.path.join(data_path, train_test_valid_file)
    vocab_pkl_path = os.path.join(pkl_path, vocab_filename)
    response_vocab_pkl_path = os.path.join(pkl_path, response_vocab_filename)
    emb_matrix_pkl_path = os.path.join(pkl_path, emb_matrix_filename)


def get_pickle(path):
    try:
        with open(path) as f:
            data = pickle.load(f)
    except IOError as e:
        print e.message
        return None
    return data


def embMatrix_from_glovePkl(filename, vocab):
    word2vec = pickle.load(open(filename))
    vocab_size = len(vocab)
    emb_matrix = np.zeros((vocab_size, len(word2vec['the'])))
    example_vector = np.array(word2vec['the'])
    try:
        for token in vocab:
            emb_matrix[vocab[token], :] = word2vec[token] if word2vec.has_key(token) else np.zeros_like(example_vector)
    except ValueError:
        logger.info(token + "NOT IN GLOVE")
    return emb_matrix


def dump_all_dictionaries(vocab, responses_dic, word_embeddings):
    with open(Paths.response_vocab_pkl_path, 'w') as f:
        pickle.dump(responses_dic, f)
    with open(Paths.vocab_pkl_path, 'w') as f:
        pickle.dump(vocab, f)
    with open(Paths.emb_matrix_pkl_path, 'w') as f:
        pickle.dump(word_embeddings, f)
    return


def load_id2v():
    with open(Paths.emb_matrix_pkl_path) as f:
        word_emb = pickle.load(f)
    return word_emb


def gen_vocab_from_txt(flag, load_existing_vocab=False):
    def load_existing_data():
        try:
            vocab = get_pickle(Paths.vocab_pkl_path)
            response_vocab = get_pickle(Paths.response_vocab_pkl_path)
            return vocab, response_vocab
        except IOError as e:
            print(e.message)
            print("IOError in gen_vocab_from_txt: Dictionary may not exist, build vocab, etc. from scratch")
            exit(1)

    if (flag == Flags.TRAIN and load_existing_vocab):
        return load_existing_data()
    elif (flag == Flags.TRAIN):
        response_vocab, vocab = {}, {}
        num_classes = 0
        vocab_set = set()

        for i, line in enumerate(open(Paths.train_test_valid_path)):
            target_label, utterance = line.lower().strip().split('\t')
            if (target_label not in response_vocab.keys()):
                response_vocab[target_label] = num_classes
                num_classes += 1
            words = utterance.split()
            for word in words:
                vocab_set.add(word)
        for word_id, word in enumerate(vocab_set):
            vocab[word] = word_id
        return vocab, response_vocab

    else:
        assert flag == Flags.TEST and load_existing_vocab == True
        return load_existing_data()


def preprocess_data(flag, load_existing_vocab):
    vocab, response_vocab = gen_vocab_from_txt(flag, load_existing_vocab)
    if (flag == Flags.TRAIN):
        path = Paths.train_path
    else:
        path = Paths.test_path

    if (not load_existing_vocab):
        emb_matrix = embMatrix_from_glovePkl(Paths.glove_path, vocab)
        dump_all_dictionaries(vocab, response_vocab, emb_matrix)
    else:
        emb_matrix = load_id2v()
    targets, utterances = [], []
    valid_targets, valid_utterances = [], []

    def get_from_vocab(arg):
        if (vocab.has_key(arg)):
            return vocab[arg]
        else:
            return 0

    for line in open(path):
        try:
            target_label, utterance = line.lower().strip().split('\t')
        except:
            logger.debug(" COULD NOT PROCESS THE LINE: " + line)
        targets.append(response_vocab[target_label])
        token_ids = map(get_from_vocab, utterance.split())
        utterances.append(token_ids)

    for line in open(Paths.valid_path):
        try:
            target_label, utterance = line.lower().strip().split('\t')
        except:
            logger.debug(" COULD NOT PROCESS THE LINE: " + line)
        valid_targets.append(response_vocab[target_label])
        token_ids = map(get_from_vocab, utterance.split())
        valid_utterances.append(token_ids)
    num_ex = len(utterances)
    print num_ex
    return valid_utterances, valid_targets, utterances, targets, vocab, response_vocab, emb_matrix

# preprocess_data(Flags.TRAIN, False)
