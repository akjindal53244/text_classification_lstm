import os
import preprocess
import cPickle as pickle

def split_data(filename, partition, responses_dic):
    with open(filename) as f:
        data = f.readlines()
    classwise_dictionary ={x:[] for x in responses_dic}

    for line in data:
        label, utterance = line.lower().split('\t')
        try:
            classwise_dictionary[label].append(utterance)
        except KeyError as e:
            print e.message
    valid_dictionary, train_dictionary={},{}
    for label in classwise_dictionary:
        num_ex = len(classwise_dictionary[label])
        valid_dictionary[label] = classwise_dictionary[label][int(partition * num_ex) : ]
        train_dictionary[label] = classwise_dictionary[label][ : int(partition * num_ex)]
    with open(preprocess.Paths.train_path,'w') as f:
        for label in train_dictionary:
            for line in train_dictionary[label]:
                f.write(label+'\t'+ line)
    with open(preprocess.Paths.valid_path,'w') as f:
        for label in valid_dictionary:
            for line in valid_dictionary[label]:
                f.write(label+'\t'+ line)
    return

if __name__ =="__main__":
    os.chdir('../')
    responses_dictionary = pickle.load(open(preprocess.Paths.response_vocab_pkl_path))
    split_data('data/train2.txt', 0.9, responses_dictionary)