import nengo
import numpy as np
from data_utils import *
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
img1 = mnist.train.images[0]

model = nengo.Network()
with model:
    my_spikes = encode(img1)
    stim = nengo.Node(my_spikes)
    a = nengo.Ensemble(n_neurons=784, dimensions=1)
    b = nengo.Ensemble(n_neurons=1000, dimensions=1)
    output = nengo.Node(output=callable, size_in=1, size_out=1)
    nengo.Connection(stim, a.neurons)
    nengo.Connection(a, b)
    nengo.Connection(b,output)
    

	
	#nengo.Connection(input_node, ens.neurons, synapse=None, transform=nengo_dl.dists.Glorot())
	#nengo.Connection(ens.neurons, output, synapse=None, transform=nengo_dl.dists.Glorot())

#with nengo_dl.Simulator(net) as sim:
	#sim.train({input_node: spike_train}, {output_p, spike_train})
