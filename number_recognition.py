import nengo
import numpy as np
from data_utils import *
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def sparsity_measure(vector):  # Gini index
    # Max sparsity = 1 (single 1 in the vector)
    v = np.sort(np.abs(vector))
    n = v.shape[0]
    k = np.arange(n) + 1
    l1norm = np.sum(v)
    summation = np.sum((v / l1norm) * ((n - k + 0.5) / n))
    return 1 - 2 * summation


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
img1 = mnist.train.images[0]

#follows the default model of creating a Nengo SNN
model = nengo.Network()
with model:
#Used to input data from spike trains generated from Data_utils and MNIST data
    my_spikes = encode(img1)
#Cast input into the Nodes to be used as input
    stim = nengo.Node(my_spikes)
#Create the layer of input that takes in the data from my_spikes
    a = nengo.Ensemble(n_neurons=784, dimensions=1)
    #b = nengo.Ensemble(n_neurons=1000, dimensions=1)
#Create layer for output
    output = nengo.Ensemble(n_neurons=10, dimensions=1)
    # output = nengo.Node(output=callable, size_in=1, size_out=10)
#Connections made between the input neurons, and the output neurons to the trainer
    nengo.Connection(stim, a.neurons)
    #nengo.Connection(a, b)
    #nengo.Connection(b,output)
    conn = nengo.Connection(a,output,solver=nengo.solvers.LstsqL2(weights=True))
    conn.learning_rule_type = nengo.Oja(learning_rate=6e-8)

    pre_p = nengo.Probe(a, synapse=0.01)
    post_p = nengo.Probe(output, synapse=0.01)
    weights_p = nengo.Probe(conn, 'weights', synapse=0.01, sample_every=0.01)

with nengo.Simulator(model) as sim:
    sim.run(20.0)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(sim.trange(), sim.data[pre_p], label="Pre")
    #plt.plot(sim.trange(), sim.data[post_p], label="Post")
    #plt.ylabel("Decoded value")
    plt.ylim(-1.6, 1.6)
    #plt.legend(loc="lower left")
    #plt.subplot(2, 1, 2)
    # Find weight row with max variance
    neuron = np.argmax(np.mean(np.var(sim.data[weights_p], axis=0), axis=1))
    #plt.plot(sim.trange(), sim.data[weights_p][..., neuron])
    #plt.ylabel("Connection weight")
    #print(
    #    "Starting sparsity: {0}".format(sparsity_measure(sim.data[weights_p][0])))
    #print("Ending sparsity: {0}".format(sparsity_measure(sim.data[weights_p][-1])))
