####################
##### We'll be defining our multilayer perceptron as a series of "layers", 
##### each applied successively to the input to produce the network output. Each 
##### layer is defined as a class, which stores a weight matrix and a bias vector
##### and includes a function for computing the layer's output.
########################
'''
A layer of a neural network, computes s(Wx + b) where s is a nonlinearity and x is the input vector.
:parameters:
- W_init : np.ndarray, shape=(n_output, n_input)
Values to initialize the weight matrix to.
- b_init : np.ndarray, shape=(n_output,)
Values to initialize the bias vector
- activation : theano.tensor.elemwise.Elemwise
Activation function for layer output
'''
from __future__ import print_function
import numpy as np
import matplotlib as plt
import theano
import theano.tensor as T
class Layer(object):
	def __init__(self, W_init, b_init, activation):
		n_output, n_input = W_init.shape
		# Make sure b is n_output in size
		assert b_init.shape == (n_output,)
		self.W = theano.shared(value=W_init.astype(theano.config.floatX),name='W', borrow=True) #borrow true just and opt
		self.b = theano.shared(value=b_init.reshape(n_output, 1).astype(theano.config.floatX), name='b',borrow=True,broadcastable=(False, True)) # broadcastable is an opt
		self.activation = activation
		self.params = [self.W, self.b] # we compute the gradient respect to those parameter
	
	
	'''
	Compute this layer's output given an input
	:parameters:
	- x : theano.tensor.var.TensorVariable
	Theano symbolic variable for layer input
	:returns:
	- output : theano.tensor.var.TensorVariable
	Mixed, biased, and activated x
	'''
	
	def output(self, x):
		# Compute linear mix
		lin_output = T.dot(self.W, x) + self.b
		# Output is just linear mix if no activation function
		# Otherwise, apply the activation function
		if(self.activation is None):
			return lin_output
		else:
			return self.activation(lin_output)