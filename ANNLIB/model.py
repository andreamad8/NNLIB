from __future__ import print_function
import numpy as np
import matplotlib as plt
import theano
import theano.tensor as T
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from numpy import linalg as LA
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import euclidean_distances
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.regularizers import l2, activity_l2
from .layer import Layer
from .MLP import MLP
from .NN import NN

#plt.rc('text', usetex=True)
#plt.rc('font', family='Times-Roman')
sns.set_style(style='white')

class Model(object):
	def __init__(self,X_train,y_train,X_validation,y_validation,X_test,y_test):
		self.X_train=X_train
		self.y_train=y_train
		self.X_validation=X_validation
		self.y_validation=y_validation
		self.X_test=X_test
		self.y_test=y_test
		self.loss_training=[]
		self.acc_training=[]
		self.loss_val=[]
		self.acc_val=[]
		self.learning_rate=0.00
		self.momentum=0.00
		self.lamb=0.00
		self.activations=0.00
		self.accuracy_train=0.0
		self.accuracy_val=0.0
		self.accuracy_test=0.0
		self.cost_train=0.0
		self.cost_val=0.0

	def LinerReg(self):
		'''
		Linar model using scikit_learn
		'''
		regr = linear_model.LinearRegression()
		regr.fit(self.X_train.transpose(), self.y_train.transpose())
		out=regr.predict(self.X_test.transpose())
		print("Mean Euclidian Error: %.2f"% self.accuracy_regression(out,self.y_test.transpose()))


	def KerasANN(self,hidden_unit,learning_rate,momentum,lamb):
		'''
		Keras ANN model, run the same model 10 times and takes the average
		'''
		temp=[]
		for i in range(10):
			model = Sequential()
			model.add(Dense(hidden_unit, input_dim=10,W_regularizer=l2(lamb), init='normal', activation='sigmoid'))
			model.add(Dense(2, init='normal', W_regularizer=l2(lamb),activation='linear'))
			# Compile model
			sgd = SGD(lr=learning_rate, momentum=momentum)
			model.compile(loss='mean_squared_error', optimizer=sgd,metrics=['mean_squared_error'])
			#print(X_t.shape,y_t.shape)
			#print(model.summary())
			for i in range(8000):
				model.train_on_batch(self.X_train.transpose(),self.y_train.transpose())
			scores = model.evaluate(self.X_test.transpose(),self.y_test.transpose(), verbose=0)
			prediction=model.predict_on_batch(self.X_test.transpose())
			MEE=self.accuracy_regression(prediction,self.y_test)
			temp.append([scores[1],MEE])
		a=np.array(temp)
		print(np.mean(a, axis=0))
		print(np.std(a,axis=0))

		print(MEE)


	def ANNModel(self,hidden_unit,outputsize,learning_rate,momentum,lamb,activations,loss):
		'''
		init the NN model, with the selected parameters
		'''
		self.learning_rate=learning_rate
		self.momentum=momentum
		self.lamb=lamb
		self.activations=activations
		layer_sizes = [self.X_train.shape[0], hidden_unit, outputsize]
		self.model=NN(self.X_train,self.y_train,layer_sizes,learning_rate,momentum,lamb,activations,loss)

	def accuracy_binary(self,output,y):
		'''
		accuracy used for monk
		'''
		return np.mean((output > .5) == y)

	def accuracy_regression(self,output,y):
		'''
		MEE calculator, sometimes we need to change transpose if we use other models
		'''
		output=output.transpose()
		y=y.transpose()
		temp=0.0
		for i in range(max(y.shape)):
			temp+=LA.norm(output[i]-y[i])
		return temp/max(y.shape)


	def train(self,max_iteration):
		'''
		function used to train our model
		'''
		iteration = 0
		cost = 0
		#print("TRAINING STARTED \n")
		while iteration < max_iteration:
			##training
			current_cost = self.model.train(self.X_train, self.y_train)
			current_output = self.model.mlp_output(self.X_train)
			cost_train=self.model.cost_function(self.X_train, self.y_train)
			self.accuracy_train = self.accuracy_regression(current_output,self.y_train)
			self.acc_training.append(self.accuracy_train)
			self.loss_training.append(float(cost_train))

			##validation
			current_output_val = self.model.mlp_output(self.X_validation)
			cost_val=self.model.cost_function(self.X_validation, self.y_validation)
			self.accuracy_val = self.accuracy_regression(current_output_val,self.y_validation)
			self.acc_val.append(self.accuracy_val)
			self.loss_val.append(float(cost_val))
			if(iteration%500==0):
				print('Loss traning   : {:.3f}, Accuracy traning   : {:.3f}'.format(float(current_cost), self.accuracy_train))
				#print('Loss validation: {:.3f}, Accuracy validation: {:.3f}'.format(float(cost_val),float(self.accuracy_val)))
			iteration += 1


		##TO RUN CROSS VALIDATION
		'''
		current_output = self.model.mlp_output(self.X_train)
		self.cost_train=self.model.cost_function(self.X_train, self.y_train)
		self.accuracy_train = self.accuracy_regression(current_output,self.y_train)
		current_output_val = self.model.mlp_output(self.X_validation)
		self.cost_val=self.model.cost_function(self.X_validation, self.y_validation)
		self.accuracy_val = self.accuracy_regression(current_output_val,self.y_validation)
		'''
		self.iter=iteration
		#print('Loss traning   : {:.3f}, Accuracy traning   : {:.3f}'.format(float(current_cost), self.accuracy_train))
		#print('Loss validation: {:.3f}, Accuracy validation: {:.3f}'.format(float(cost_val),float(self.accuracy_val)))



	def predict(self, X_blindtest):
		'''
		give X as input predict the output using the previous model
		'''
		return self.model.mlp_output(X_blindtest)

	def test(self,X_test,y_test):
		'''
		evalute the model using the test datas
		'''
		output = self.model.mlp_output(self.X_test)
		cost_test=self.model.cost_function(self.X_test, self.y_test)
		self.accuracy_test = self.accuracy_regression(output,self.y_test)
		print('Loss test      : {:.3f}, Accuracy test      : {:.3f}'.format(float(cost_test),float(self.accuracy_test)))
		print("\n")



	def plotLA(self):
		'''
		to generate the plots of the loss
		'''
		fig = plt.figure()
		ax1 = fig.add_subplot(2, 1, 1)
		#inset1 = inset_axes(ax1, width="60%", height="60%")

		ax1.plot(self.loss_training, label="Training",linewidth=1.95, alpha=0.7, color='red')
		ax1.plot(self.loss_val,label="Test",color="gray",linewidth=1.95, alpha=0.7, linestyle='dashed')
		ax1.annotate('Learning rate: %s'% str(self.learning_rate),xy=(0.6,0.9), xycoords='axes fraction',size=13)
		ax1.annotate('Momentum: %s'%  str(self.momentum),xy=(0.6,0.8), xycoords='axes fraction',size=13)
		ax1.annotate('L2: %s'%  str(self.lamb),xy=(0.6,0.7), xycoords='axes fraction',size=13)
		ax1.set_ylabel("MSE")
		ax1.yaxis.grid(True)
		ax1.set_xlim([0,self.iter+1 ])
		plt.title("MONK")
		'''
		inset1.plot(self.loss_training, label="Training",linewidth=1.95, alpha=0.7, color='red')
		inset1.plot(self.loss_val,label="Test",color="gray",linewidth=1.95, alpha=0.7, linestyle='dashed')
		inset1.set_xlim([0, 60])
		inset1.yaxis.grid(True)
		'''
		ax1.legend(loc=1)
		ax2 = fig.add_subplot(2, 1, 2)
		#inset2 = inset_axes(ax2, width="60%", height="60%")

		ax2.plot(self.acc_training,label="Training",linewidth=1.95, alpha=0.7, color='red')
		ax2.plot(self.acc_val,label="Test",color="gray",linewidth=1.95, alpha=0.7, linestyle='dashed')
		'''
		inset2.plot(self.acc_training,label="Training",linewidth=1.95, alpha=0.7, color='red')
		inset2.plot(self.acc_val,label="Test",color="gray",linewidth=1.95, alpha=0.7, linestyle='dashed')
		inset2.set_xlim([0, 60])
		inset2.yaxis.grid(True)
		'''
		ax2.set_xlabel("Epochs")
		ax2.set_ylabel("Accuracy")
		ax2.set_xlim([0,self.iter+1 ])
		ax2.yaxis.grid(True)
		#plt.savefig('IMG/AA1CUP.png', format='png',dpi=400)
		plt.show()
