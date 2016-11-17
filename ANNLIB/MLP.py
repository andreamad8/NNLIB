##################################
##Most of the functionality of our MLP is contained in the Layer class; the MLP class is essentially just a container for a list of Layers and their parameters. The output
##function simply recursively computes the output for each layer. Finally, the squared_error returns the squared Euclidean distance between the output of the network given
##an input and the desired (ground truth) output. This function is meant to be used as a cost in the setting of minimizing cost over some training data. As above, the output
##and squared error functions are not to be used for actually computing values; instead, they're to be used to create functions which are used to compute values.
##################################

'''
Multi-layer perceptron class, computes the composition of a sequence of Layers
:parameters:
- W_init : list of np.ndarray, len=N
Values to initialize the weight matrix in each layer to.
The layer sizes will be inferred from the shape of each matrix in W_init
- b_init : list of np.ndarray, len=N
Values to initialize the bias vector in each layer to
- activations : list of theano.tensor.elemwise.Elemwise, len=N
Activation function for layer output for each layer
'''
from __future__ import print_function
import numpy as np
import matplotlib as plt
import theano
import theano.tensor as T
from .layer import Layer
class MLP(object):
	def __init__(self, W_init, b_init, activations):
		# Make sure the input lists are all of the same length
		assert len(W_init) == len(b_init) == len(activations)

		self.layers = []
		for W, b, activation in zip(W_init, b_init, activations):
			self.layers.append(Layer(W, b, activation))

		self.params = []
		for layer in self.layers:
			self.params += layer.params
		self.L2_sqr = (self.params[0] ** 2).sum() + (self.params[2] ** 2).sum()



	def output(self, x):
		# Recursively compute output
		for layer in self.layers:
			x = layer.output(x)
		return x

	'''
	Compute the squared euclidean error of the network output against the "true" output y
	- x : theano.tensor.var.TensorVariable
	- y : theano.tensor.var.TensorVariable
	returns
	- error : theano.tensor.var.TensorVariable
	'''

	def mean_squared_error(self, x, y,L2):
		cost=T.mean((self.output(x) - y)**2)+L2*self.L2_sqr
		return cost

	def squared_error(self, x, y,L2):
		cost=T.sum((self.output(x) - y)**2)+L2*self.L2_sqr
		return cost

