from __future__ import print_function
import numpy as np
import matplotlib as plt
import theano
import theano.tensor as T
class Layer(object):
	'''
	A layer of a neural network, computes s(Wx + b) where s is a nonlinearity and x is the input vector.
	:parameters:
	- W_init : multidimentional numpy array
	- b_init : numpy array
	- activation : theano elemwise funtion or null in case of linear output
	'''
	def __init__(self, W_init, b_init, activation):
		n_output, n_input = W_init.shape
		# Make sure b is n_output in size
		assert b_init.shape == (n_output,)
		self.W = theano.shared(value=W_init.astype(theano.config.floatX),name='W', borrow=True) #borrow true just and opt
		self.b = theano.shared(value=b_init.reshape(n_output, 1).astype(theano.config.floatX), name='b',borrow=True,broadcastable=(False, True)) # broadcastable is an opt
		self.activation = activation
		self.params = [self.W, self.b] # we compute the gradient respect to those parameter

	def output(self, x):
		'''
		Compute this layer's output given an input
		:parameters:
		- x : Symbolic theano variable
		:returns:
		- output : Symbolic theano variable s(Wx + b)
		 s can be also an identity function
		'''
		lin_output = T.dot(self.W, x) + self.b
		if(self.activation is None):
			return lin_output
		else:
			return self.activation(lin_output)
