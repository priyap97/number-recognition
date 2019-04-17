import nengo
import numpy as np
import matplotlib.pyplot as plt


model = nengo.Network()
with model:
#	my_spikes = 
#	input_node = nengo.Node(nengo.processes.PresentInput(my_spikes, 0.001))  use this function, my_spikes is calling Nikita's function
	stim = nengo.Node([0]) #This is the input, change this using Nikita's function
	a = nengo.Ensemble(n_neurons=1000, dimensions=1)
	b = nengo.Ensemble(n_neurons=10, dimensions=1)
	output = nengo.Node(output=callable, size_in=1, size_out=1)
	nengo.Connection(stim, a)
	nengo.Connection(a, b)
	nengo.Connection(b,output)
    

	
	#nengo.Connection(input_node, ens.neurons, synapse=None, transform=nengo_dl.dists.Glorot())
	#nengo.Connection(ens.neurons, output, synapse=None, transform=nengo_dl.dists.Glorot())

#with nengo_dl.Simulator(net) as sim:
	#sim.train({input_node: spike_train}, {output_p, spike_train})
