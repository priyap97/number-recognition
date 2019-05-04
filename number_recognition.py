import nengo
import numpy as np
from data_utils import *
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
testing_images = mnist.train.images
testing_encoded_data = list()
for image in testing_images:
    testing_encoded_data.append(encode(image))

#follows the default model of creating a Nengo SNN
model = nengo.Network()
with model:
    #Cast input into the Nodes to be used as input
    stim = nengo.Node(nengo.processes.PresentInput(testing_encoded_data, 0.1))

    #Create the layer of input that takes in the data from my_spikes
    a = nengo.Ensemble(n_neurons=784, dimensions=1)

    # Create hidden layer
    b = nengo.Ensemble(n_neurons=1000, dimensions=1)

    #Create layer for output
    output = nengo.Ensemble(n_neurons=10, dimensions=1)

    #Connections made between the input neurons, and the output neurons to the trainer
    nengo.Connection(stim, a.neurons)

    conn_ab = nengo.Connection(a,b,solver=nengo.solvers.LstsqL2(weights=True))
    conn_ab.learning_rule_type = nengo.Oja(learning_rate=6e-8)

    conn_boutput = nengo.Connection(b, output, solver=nengo.solvers.LstsqL2(weights=True))
    conn_boutput.learning_rule_type = nengo.Oja(learning_rate=6e-8)

    pre_p = nengo.Probe(a, synapse=0.01)
    post_p = nengo.Probe(output, synapse=0.01)
    weights_p = nengo.Probe(conn, 'weights', synapse=0.01, sample_every=0.01)
