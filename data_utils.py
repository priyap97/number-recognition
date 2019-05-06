import gzip
import pickle
import numpy


def get_input_data():
    filename = 'mnist.pkl.gz'
    decomp = gzip.open(filename, 'rb').read()
    train_set, valid_set, test_set = pickle.loads(decomp, encoding='latin1')
    train_set = list(train_set)
    test_set = list(test_set)
    for data in (train_set, test_set):
        one_hot = numpy.zeros((data[0].shape[0], 10))
        one_hot[numpy.arange(data[0].shape[0]), data[1]] = 1
        data[1] = one_hot
    return train_set
