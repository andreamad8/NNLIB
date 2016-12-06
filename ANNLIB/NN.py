from __future__ import print_function
import numpy as np
import matplotlib as plt
import theano
import theano.tensor as T
import theano.d3viz as d3v
import math
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


	########################
	### NOTE: this function is called just one, then theano create a compiled version
	### and then we need to init the shared variable. For instance for the momentum we
	### need to init the variable that keeps track of the previous grad updates
	def gradient_updates(self,cost, params, learning_rate, momentum):
		assert momentum < 1 and momentum >= 0
		#assert learning_rate < 1
		assert self.lamb<1
		updates = []
		if(momentum==0):
			for param in params:
				updates.append((param, param - learning_rate*T.grad(cost, param)))
		else:
			for param in params:
				#init param_update to keep track of previous changes
				param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
				# first implicit init param = param - learing_rate*0
				# then by the second time on param_update is going to be a*oldGrad + (1-a) newGrad
				updates.append((param, param - learning_rate*param_update))
				updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
		return updates

	def initNN(self):
		W_init = []
		b_init = []
		activations = []
		
		for n_input, n_output in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
			#W_init.append( np.asarray(np.random.uniform(size=(n_output, n_input),low=-.7, high=.7), dtype=theano.config.floatX))
			W_init.append( np.random.randn(n_output, n_input) * math.sqrt(2.0/n_input))
			
			b_init.append(np.zeros(n_output))

			if(self.activations == "sigmoid"):
				activations.append(T.nnet.sigmoid)
			if(self.activations == "regression"):
				activations.append(T.nnet.sigmoid)
				activations.append(None)
				# notice when is none the layer has as default the dot prod Wx +b
				self.activations = "none"

		#create MLP
		mlp = MLP(W_init, b_init, activations)
		mlp_input = T.matrix('mlp_input')
		mlp_target = T.matrix('mlp_target')
		L2 = T.scalar('L2')
		if(self.loss=="MSE"):
			cost = mlp.mean_squared_error(mlp_input, mlp_target,L2)
		elif(self.loss=="SE"):
			cost = mlp.squared_error(mlp_input, mlp_target,L2)

		
		self.train = theano.function([mlp_input, mlp_target, theano.In(L2, value=self.lamb)], cost, updates=self.gradient_updates(cost, mlp.params, self.learning_rate,self.momentum))
		## utility function to check the loss 
		self.cost_function = theano.function([mlp_input, mlp_target, theano.In(L2, value=self.lamb)], cost)
		# Create a theano function for computing the MLP's output given some input
		self.mlp_output = theano.function([mlp_input], mlp.output(mlp_input))

	def plot_computational_graph(self,path):
		theano.printing.pydotprint(self.mlp_output, outfile=path, var_with_name_simple=True)
		theano.printing.pydotprint(self.cost_function, outfile=path, var_with_name_simple=True)
		theano.printing.pydotprint(self.train, outfile=path, var_with_name_simple=True)


