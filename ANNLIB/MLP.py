from __future__ import print_function
import numpy as np
import matplotlib as plt
import theano
import theano.tensor as T
from .layer import Layer
class MLP(object):
	'''
	MLP class, computes the composition of a sequence of Layers, notice that also
	this layer is symbolic.
	:parameters:
	- W_init : multidimentional numpy array
	- b_init : numpy array
	- activations : list of theano elemwise funtion
	'''
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


	def mean_squared_error(self, x, y,L2):
		'''
		- x : Symbolic theano variable
		- y : Symbolic theano variable
		returns
		- MSE : Theano function ( composition of variable)
		'''
		cost=T.mean(T.square(self.output(x) - y))+L2*self.L2_sqr
		return cost

	def squared_error(self, x, y,L2):
		'''
		- x : Symbolic theano variable
		- y : Symbolic theano variable
		returns
		- SE : Theano function ( composition of variable)
		'''
		cost=T.sum((self.output(x) - y)**2)+L2*self.L2_sqr
		return cost
