from __future__ import print_function
import numpy as np
import matplotlib as plt
import theano
import theano.tensor as T
#import seaborn
from ANNLIB.MLP import MLP 

'''
Compute updates for gradient descent with momentum
:parameters:
- cost : theano.tensor.var.TensorVariable
Theano cost function to minimize
- params : list of theano.tensor.var.TensorVariable
Parameters to compute gradient against
- learning_rate : float
Gradient descent learning rate
- momentum : float
Momentum parameter, should be at least 0 (standard gradient descent) and less than 1
:returns:
updates : list
List of updates, one for each parameter
'''

def gradient_updates_momentum(cost, params, learning_rate, momentum):
	assert momentum < 1 and momentum >= 0
	updates = []

	for param in params:
		# For each parameter, we'll create a param_update shared variable.
		# This variable will keep track of the parameter's update step across iterations.
		# We initialize it to 0
		param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
		# Each parameter is updated by taking a step in the direction of the gradient.
		# However, we also "mix in" the previous step according to the given momentum value.
		# Note that when updating param_update, we are using its old value and also the new gradient step.
		updates.append((param, param - learning_rate*param_update))
		# Note that we don't need to derive backpropagation to compute updates - just use T.grad!
		updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
	return updates


def gradient_updates_stocastic(cost, params, learning_rate):
	updates = []

	for param in params:
		# For each parameter, we'll create a param_update shared variable.
		# This variable will keep track of the parameter's update step across iterations.
		# We initialize it to 0
		param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
		# Each parameter is updated by taking a step in the direction of the gradient.
		# However, we also "mix in" the previous step according to the given momentum value.
		# Note that when updating param_update, we are using its old value and also the new gradient step.
		updates.append((param, param - learning_rate*param_update))
		# Note that we don't need to derive backpropagation to compute updates - just use T.grad!
		updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
	return updates

def create_data():
	# Training data - two randomly-generated Gaussian-distributed clouds of points in 2d space
	np.random.seed(0)
	# Number of points
	N = 1000
	# Labels for each cluster
	y = np.random.random_integers(0, 1, N)
	# Mean of each cluster
	means = np.array([[-1, 1], [-1, 1]])
	# Covariance (in X and Y direction) of each cluster
	covariances = np.random.random_sample((2, 2)) + 1
	# Dimensions of each point
	X = np.vstack([np.random.randn(N)*covariances[0, y] + means[0, y],np.random.randn(N)*covariances[1, y] + means[1, y]]).astype(theano.config.floatX)
	# Convert to targets, as floatX
	y = y.astype(theano.config.floatX)
	return [X,y]

def ploty(X,y):
	# Plot the data
	plt.scatter(X[0, :], X[1, :], c=y, lw=.3, s=5)
	plt.show()

if __name__=='__main__':
	Dataset= create_data()
	X=Dataset[0]
	y=Dataset[1]
	#ploty(X,y)

	# First, set the size of each layer (and the number of layers)
	# Input layer size is training data dimensionality (2)
	# Output size is just 1-d: class label - 0 or 1
	# Finally, let the hidden layers be twice the size of the input.
	# If we wanted more layers, we could just add another layer size to this list.
	layer_sizes = [X.shape[0], X.shape[0]*2, 1]
	print(layer_sizes)
	# Set initial parameter values
	W_init = []
	b_init = []
	activations = []
	for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
		#init weight
		W_init.append(np.random.randn(n_output, n_input))
		# Set initial biases to 1
		b_init.append(np.ones(n_output))
		#set the activation for all the layer => signmoid because is binary
		activations.append(T.nnet.sigmoid)

	# Create an instance of the MLP class
	mlp = MLP(W_init, b_init, activations)
	# Create Theano variables for the MLP input
	mlp_input = T.matrix('mlp_input')
	# ... and the desired output
	mlp_target = T.vector('mlp_target')
	# Learning rate and momentum hyperparameter values
	# Again, for non-toy problems these values can make a big difference
	# as to whether the network (quickly) converges on a good local minimum.
	learning_rate = 0.01
	momentum = 0.9
	# Create a function for computing the cost of the network given an input
	cost = mlp.squared_error(mlp_input, mlp_target)
	# Create a theano function for training the network
	train = theano.function([mlp_input, mlp_target], cost, updates=gradient_updates_momentum(cost, mlp.params, learning_rate, momentum))
	# Create a theano function for computing the MLP's output given some input
	mlp_output = theano.function([mlp_input], mlp.output(mlp_input))
