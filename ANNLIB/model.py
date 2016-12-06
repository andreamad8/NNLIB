from __future__ import print_function
import numpy as np
import matplotlib as plt
import theano
import theano.tensor as T
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from numpy import linalg as LA
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import euclidean_distances
from .layer import Layer
from .MLP import MLP
from .NN import NN

plt.rc('text', usetex=True)
plt.rc('font', family='Times-Roman')
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
		# Create linear regression object
		regr = linear_model.LinearRegression()
		regr.fit(self.X_train.transpose(), self.y_train.transpose())
		out=regr.predict(self.X_test.transpose())
		print("Mean Euclidian Error: %.2f"% self.accuracy_regression(out,self.y_test.transpose()))
		
	def SVR(self):
		svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,param_grid={"C": [1e0, 1e1, 1e2, 1e3],"gamma": np.logspace(-2, 2, 5)})
		svr1=svr.fit(self.X_train.transpose(), self.y_train[0].transpose())
		svr2=svr.fit(self.X_train.transpose(), self.y_train[1].transpose())
		y_rbf1 = svr1.predict(self.X_test.transpose())
		y_rbf2 = svr2.predict(self.X_test.transpose())
		y_predicted=np.array([y_rbf1,y_rbf2])
		print("Mean Euclidian Error: %.2f"% self.accuracy_regression(y_predicted.transpose(),self.y_test.transpose()))

		
	def Kridge(self):
		clf = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],"gamma": np.logspace(-2, 2, 5)})
		clf_m=clf.fit(self.X_train.transpose(), self.y_train.transpose())
		y_clf = clf_m.predict(self.X_test.transpose())
		print("Mean Euclidian Error: %.2f"% self.accuracy_regression(y_clf,self.y_test.transpose()))

	def ANNModel(self,hidden_unit,outputsize,learning_rate,momentum,lamb,activations,loss):
		self.learning_rate=learning_rate
		self.momentum=momentum
		self.lamb=lamb
		self.activations=activations
		layer_sizes = [self.X_train.shape[0], hidden_unit, outputsize]
		
		self.model=NN(self.X_train,self.y_train,layer_sizes,learning_rate,momentum,lamb,activations,loss)

	def accuracy_binary(self,output,y):
		return np.mean((output > .5) == y)

	def accuracy_regression(self,output,y):
		#print(y.shape,y.shape[0],y.shape[1])
		#return mean_squared_error(output, y)
		output=output.transpose()
		y=y.transpose()
		temp=0.0
		for i in range(max(y.shape)):
			temp+=LA.norm(output[i]-y[i])
		return temp/max(y.shape)
		

	def train(self,max_iteration):
		iteration = 0
		print("TRAINING STARTED \n")
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
			if(iteration%50==0):
				print('Loss traning   : {:.3f}, Accuracy traning   : {:.3f} \r'.format(float(current_cost), self.accuracy_train),end=' ')
			iteration += 1
		
		##TO RUN CROSS VALIDATION
		###current_output = self.model.mlp_output(self.X_train)
		###self.cost_train=self.model.cost_function(self.X_train, self.y_train)
		###self.accuracy_train = self.accuracy_regression(current_output,self.y_train)
		###current_output_val = self.model.mlp_output(self.X_validation)
		###self.cost_val=self.model.cost_function(self.X_validation, self.y_validation)
		###self.accuracy_val = self.accuracy_regression(current_output_val,self.y_validation)

		self.iter=iteration
		print('Loss traning   : {:.3f}, Accuracy traning   : {:.3f}'.format(float(current_cost), self.accuracy_train))
		print('Loss validation: {:.3f}, Accuracy validation: {:.3f}'.format(float(cost_val),float(self.accuracy_val)))


	def test(self,X_test,y_test):
		output = self.model.mlp_output(self.X_test)
		cost_test=self.model.cost_function(self.X_test, self.y_test)
		self.accuracy_test = self.accuracy_regression(output,self.y_test)
		print('Loss test      : {:.3f}, Accuracy test      : {:.3f}'.format(float(cost_test),float(self.accuracy_test)))
		print("\n")


		
	def plotLA(self):
		fig = plt.figure()
		ax1 = fig.add_subplot(2, 1, 1)
		plt.plot(self.loss_training, label="Training",linewidth=1.95, alpha=0.7, color='red')
		plt.plot(self.loss_val,label="Validation",color="gray",linewidth=1.95, alpha=0.7, linestyle='dashed')
		plt.annotate('Learning rate: %s'% str(self.learning_rate),xy=(0.6,0.9), xycoords='axes fraction',size=13)
		plt.annotate('Momentum: %s'%  str(self.momentum),xy=(0.6,0.8), xycoords='axes fraction',size=13)
		plt.annotate('L2: %s'%  str(self.lamb),xy=(0.6,0.7), xycoords='axes fraction',size=13)
		ax1.set_ylabel("Loss")
		ax1.yaxis.grid(True)
		ax1.set_xlim([0,self.iter+1 ])
		#ax1.set_xlim([0,100 ])
		#ax1.set_yscale('log')		
		plt.title("AA1 CUP")
		
		plt.legend(loc=1)
		ax2 = fig.add_subplot(2, 1, 2)
		plt.plot(self.acc_training,label="Training",linewidth=1.95, alpha=0.7, color='red')
		plt.plot(self.acc_val,label="Validation",color="gray",linewidth=1.95, alpha=0.7, linestyle='dashed')
		ax2.set_xlabel("Epochs")
		ax2.set_ylabel("MEE")
		#ax2.set_ylim([0, 1.1])
		ax2.set_xlim([0,self.iter+1 ])	
		#ax2.set_xlim([0,100 ])
				
		ax2.yaxis.grid(True)
		#plt.savefig('IMG/Monk1_%s_%s_%s.png'%(self.learning_rate,self.momentum,self.lamb), format='png')
		plt.show()

	def ROC(self):
		output = self.model.mlp_output(self.X_test)
		fpr, tpr, thresholds = metrics.roc_curve(self.y_test.reshape(-1), output.reshape(-1))
		roc_auc = auc(fpr, tpr)
		plt.figure()
		lw = 2
		plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.01])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic')
		plt.legend(loc="lower right")
		plt.show()

