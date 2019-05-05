import nengo
import os
import urllib.request as urllib


def get_input_data():
    # data_dir = nengo.rc.get('nengo_extras', 'data_dir')
    filename = 'mnist.pkl.gz'
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    # url_opener = urllib.URLopener()
    return urllib.urlretrieve(url, filename)

print(get_input_data())
