import gzip
import pickle


def get_input_data():
    filename = 'mnist.pkl.gz'
    decomp = gzip.open(filename, 'rb').read()
    train_set, valid_set, test_set = pickle.loads(decomp, encoding='latin1')
    return train_set, valid_set, test_set

print(get_input_data())
