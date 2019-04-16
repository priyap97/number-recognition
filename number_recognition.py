import nengo
import numpy as np
import matplotlib.pyplot as plt

model = nengo.Network()

with model:
	input_node = nengo.Node([0]*784)
	ens = nengo.Ensemble(1000,1) #Number of neurons, dimensions of the ensemble
	output = nengo.Node(output=callable, size_in=1, size_out=10)
	output_probe = nengo.Probe(output)

	nengo.Connection(input_node, ens.neurons, synapse=None, transform=nengo_dl.dists.Glorot())
	nengo.Connection(ens.neurons, output, synapse=None, transform=nengo_dl.dists.Glorot())

with nengo_dl.Simulator(net) as sim:
	sim.train({input_node: spike_train}, {output_p, spike_train})
