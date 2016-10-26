from __future__ import print_function
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import seaborn
from ANNLIB.MLP import MLP
from sklearn import preprocessing
import itertools
plt.rc('text', usetex=True)
plt.rc('font', family='Times-Roman')
sns.set_style(style='white')
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

def gradient_updates(cost, params, learning_rate):
	updates = []
	for param in params:
		updates.append((param, param - learning_rate*T.grad(cost, param)))
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
	return X,y

def ploty(X,y):
	# Plot the data
	plt.scatter(X[0, :], X[1, :], c=y, lw=.3, s=5)
	plt.show()

def create_Net(X,y, layer_sizes,batch,learning_rate,momentum,lamb):

		# Set initial parameter values
		W_init = []
		b_init = []
		activations = []
		#[(2L, 4L), (8L, 1)]
		for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
			W_init.append(np.random.randn(n_output, n_input))
			b_init.append(np.ones(n_output))
			activations.append(T.nnet.sigmoid)
		mlp = MLP(W_init, b_init, activations)

		mlp_input = T.matrix('mlp_input')
		mlp_target = T.vector('mlp_target')
		L2 = T.scalar('L2')


		cost = mlp.squared_error(mlp_input, mlp_target,L2)
		# Create a theano function for training the network
		if(batch):
			train = theano.function([mlp_input, mlp_target, theano.In(L2, value=lamb)], cost, updates=gradient_updates_momentum(cost, mlp.params, learning_rate,momentum))
		else:
			print("no momentum")
			train = theano.function([mlp_input, mlp_target, theano.In(L2, value=lamb)], cost, updates=gradient_updates(cost, mlp.params, learning_rate))
		cost_function = theano.function([mlp_input, mlp_target, theano.In(L2, value=lamb)], cost)
		# Create a theano function for computing the MLP's output given some input
		mlp_output = theano.function([mlp_input], mlp.output(mlp_input))
		return train, mlp_output,cost_function

def data(filename):
	with open(filename) as f:
		data = f.readlines()
	data=[e[:-1] for e in [e.split(' ') for e in  map(lambda x: x[:-1],data)]]
	translatio_2={'1':[1,0],'2':[0,1]}
	translatio_3={'1':[1,0,0],'2':[0,1,0],'3':[0,0,1]}
	translatio_4={'1':[1,0,0,0],'2':[0,1,0,0],'3':[0,0,1,0],'4':[0,0,0,1]}
	encoding={'0':translatio_3,'1':translatio_3,'2':translatio_2,'3':translatio_3,'4':translatio_4,'5':translatio_2}
	X=[ e[1:] for e in data]
	newX=[]
	for val in X:
		temp=[]
		for i in range(6):
			temp.append(encoding[str(i)][str(val[i])])
		temp=list(itertools.chain.from_iterable(temp))
		newX.append(temp)

	X=[ np.array(e, dtype=theano.config.floatX) for e in newX]

	y=[e[0] for e in data]

	X = np.array(X, dtype=theano.config.floatX)
	y = np.array(y, dtype=theano.config.floatX)

	X=X.transpose()

	return X,y

if __name__=='__main__':
	X_train,y_train=data('dataset/monks-3.train')
	X_test,y_test=data('dataset/monks-3.test')

	assert X_train.shape[0]== X_test.shape[0]
	layer_sizes = [X_train.shape[0], 4, 1]

	#ploty(X_train,y)
	train, mlp_output,cost_function=create_Net(X_train,y_train,layer_sizes,batch=True,learning_rate = 0.01,momentum = 0.5,lamb=0.01)

	iteration = 0
	max_iteration =3000
	loss_training=[]
	acc_training=[]
	loss_test=[]
	acc_test=[]
	accuracy=0
	accuracy_test=0
	while iteration < max_iteration:
		if(accuracy<1):
			##training
			current_cost = train(X_train, y_train)
			current_output = mlp_output(X_train)
			accuracy = np.mean((current_output > .5) == y_train)
			acc_training.append(accuracy)
			loss_training.append(float(current_cost))

			##test
			current_output_test = mlp_output(X_test)
			cost_test=cost_function(X_test, y_test)
			accuracy_test = np.mean((current_output_test > .5) == y_test)
			acc_test.append(accuracy_test)
			loss_test.append(float(cost_test))

			#shuffle training data
			X=X_train.transpose()
			rng_state = np.random.get_state()
			np.random.shuffle(X)
			np.random.set_state(rng_state)
			np.random.shuffle(y_train)
			X_train=X.transpose()

			if(iteration%50==0):
				print('Cost: {:.3f}, Accuracy: {:.3f}, Accuracy Test: {:.3f}'.format(float(current_cost), accuracy,accuracy_test))
			iteration += 1
		else:
			break
	print('Cost: {:.3f}, Accuracy: {:.3f}, Accuracy Test: {:.3f}'.format(float(current_cost), accuracy,accuracy_test))
	print(iteration)

	fig = plt.figure()
	ax1 = fig.add_subplot(2, 1, 1)
	plt.plot(loss_training, label="Training",linewidth=1.95, alpha=0.7, color='red')
	plt.plot(loss_test,label="Test",color="gray",linewidth=1.95, alpha=0.7, linestyle='dashed')
	ax1.set_xlabel("Epoch")
	ax1.set_ylabel("Loss")
	ax1.yaxis.grid(True)
	ax1.set_xlim([0,iteration+1 ])

	plt.legend()
	ax2 = fig.add_subplot(2, 1, 2)
	plt.plot(acc_training,label="Training",linewidth=1.95, alpha=0.7, color='red')
	plt.plot(acc_test,label="Test",color="gray",linewidth=1.95, alpha=0.7, linestyle='dashed')
	ax2.set_xlabel("Accuracy")
	ax2.set_ylabel("Loss")
	ax2.set_ylim([0, 1.1])
	ax2.set_xlim([0,iteration+1 ])

	ax2.yaxis.grid(True)

	plt.show()
