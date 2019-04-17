import nengo
import numpy as np
import data_utils.py
import matplotlib.pyplot as plt


model = nengo.Network()
with model:
	my_spikes = encode(img1, True)
	input_node = nengo.Node(nengo.processes.PresentInput(my_spikes, 0.001))
	a = nengo.Ensemble(n_neurons=1000, dimensions=1)
	b = nengo.Ensemble(n_neurons=10, dimensions=1)
	output = nengo.Node(output=callable, size_in=1, size_out=1)
	nengo.Connection(input_node, a)
	nengo.Connection(a, b)
	nengo.Connection(b,output)
    

	
	#nengo.Connection(input_node, ens.neurons, synapse=None, transform=nengo_dl.dists.Glorot())
	#nengo.Connection(ens.neurons, output, synapse=None, transform=nengo_dl.dists.Glorot())

#with nengo_dl.Simulator(net) as sim:
	#sim.train({input_node: spike_train}, {output_p, spike_train})
