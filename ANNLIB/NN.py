from __future__ import print_function
import numpy as np
import matplotlib as plt
import theano
import theano.tensor as T
import theano.d3viz as d3v
from .layer import Layer
from .MLP import MLP


class NN(object):
	def __init__(self,X,y,layer_sizes,learning_rate,momentum,lamb,activations,loss):
		self.X=X
		self.y=y
		self.layer_sizes=layer_sizes
		self.learning_rate=learning_rate
		self.momentum=momentum
		self.lamb=lamb
		self.activations=activations
		if(loss=="MSE" or loss=="SE"):
			self.loss=loss
		else:
			self.loss="MSE"
		self.initNN()



	def gradient_updates_momentum(self,cost, params, learning_rate, momentum):
		assert momentum < 1 and momentum >= 0
		updates = []
		if(momentum==0):
			for param in params:
				updates.append((param, param - learning_rate*T.grad(cost, param)))
		else:
			for param in params:
				#init for the momentum 
				param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
				updates.append((param, param - learning_rate*param_update))
				updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
		return updates


	def initNN(self):
		W_init = []
		b_init = []
		activations = []
		
		for n_input, n_output in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
			W_init.append( np.asarray(np.random.uniform(size=(n_output, n_input),low=-.4, high=.4), dtype=theano.config.floatX))
			b_init.append(np.ones(n_output))
			if(self.activations == "sigmoid"):
				activations.append(T.nnet.sigmoid)

		#create MLP
		mlp = MLP(W_init, b_init, activations)

		mlp_input = T.matrix('mlp_input')
		mlp_target = T.vector('mlp_target')
		L2 = T.scalar('L2')
		if(self.loss=="MSE"):
			cost = mlp.mean_squared_error(mlp_input, mlp_target,L2)
		elif(self.loss=="SE"):
			cost = mlp.squared_error(mlp_input, mlp_target,L2)

		
		self.train = theano.function([mlp_input, mlp_target, theano.In(L2, value=self.lamb)], cost, updates=self.gradient_updates_momentum(cost, mlp.params, self.learning_rate,self.momentum))
		## utility function to check the loss 
		self.cost_function = theano.function([mlp_input, mlp_target, theano.In(L2, value=self.lamb)], cost)
		# Create a theano function for computing the MLP's output given some input
		self.mlp_output = theano.function([mlp_input], mlp.output(mlp_input))

	def plot_computational_graph(self,path):
		theano.printing.pydotprint(self.mlp_output, outfile=path, var_with_name_simple=True)
		theano.printing.pydotprint(self.cost_function, outfile=path, var_with_name_simple=True)
		theano.printing.pydotprint(self.train, outfile=path, var_with_name_simple=True)


