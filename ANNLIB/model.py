from __future__ import print_function
import numpy as np
import matplotlib as plt
import theano
import theano.tensor as T
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='Times-Roman')
sns.set_style(style='white')

from .layer import Layer
from .MLP import MLP
from .NN import NN


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
		

	def ANNModel(self,hidden_unit,outputsize,learning_rate,momentum,lamb,activations,loss):
		layer_sizes = [self.X_train.shape[0], hidden_unit, outputsize]
		self.model=NN(self.X_train,self.y_train,layer_sizes,learning_rate,momentum,lamb,activations,loss)

	def accuracy_binary(self,output,y):
		return np.mean((output > .5) == y)

	def train(self,max_iteration):
		iteration = 0
		accuracy=0
		accuracy_val=0
		while iteration < max_iteration:	
			##training
			current_cost = self.model.train(self.X_train, self.y_train)
			current_output = self.model.mlp_output(self.X_train)
			accuracy = self.accuracy_binary(current_output,self.y_train)
			self.acc_training.append(accuracy)
			self.loss_training.append(float(current_cost))
			
			##validation
			current_output_val = self.model.mlp_output(self.X_validation)
			cost_val=self.model.cost_function(self.X_validation, self.y_validation)
			accuracy_val = self.accuracy_binary(current_output_val,self.y_validation)
			self.acc_val.append(accuracy_val)
			self.loss_val.append(float(cost_val))
			if(iteration%50==0):
				print('Cost: {:.3f}, Accuracy: {:.3f}, Accuracy validation: {:.3f} \r'.format(float(current_cost), accuracy,accuracy_val))
			iteration += 1
		self.iter=iteration
		print('Loss validation: {:.3f}, Accuracy validation: {:.3f}'.format(float(cost_val),float(accuracy_val)))


	def test(self):
		output = self.model.mlp_output(self.X_test)
		cost_test=self.model.cost_function(self.X_test, self.y_test)
		accuracy_test = self.accuracy_binary(output,self.y_test)
		print('Loss test: {:.3f}, Accuracy test: {:.3f}'.format(float(cost_test),float(accuracy_test)))

		
	def plotLA(self):
		fig = plt.figure()
		ax1 = fig.add_subplot(2, 1, 1)
		plt.plot(self.loss_training, label="Training",linewidth=1.95, alpha=0.7, color='red')
		plt.plot(self.loss_val,label="Test",color="gray",linewidth=1.95, alpha=0.7, linestyle='dashed')
		#ax1.set_xlabel("Epochs")
		ax1.set_ylabel("Loss")
		ax1.yaxis.grid(True)
		ax1.set_xlim([0,self.iter+1 ])

		plt.legend()
		ax2 = fig.add_subplot(2, 1, 2)
		plt.plot(self.acc_training,label="Training",linewidth=1.95, alpha=0.7, color='red')
		plt.plot(self.acc_val,label="Test",color="gray",linewidth=1.95, alpha=0.7, linestyle='dashed')
		ax2.set_xlabel("Epochs")
		ax2.set_ylabel("Accuracy")
		ax2.set_ylim([0, 1.1])
		ax2.set_xlim([0,self.iter+1 ])

		ax2.yaxis.grid(True)

		plt.show()