from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
from ANNLIB.model import Model
from sklearn import preprocessing
import itertools
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
import csv
import math
import itertools as it
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.regularizers import l2, activity_l2
from numpy import linalg as LA
import csv

### PARAMETER FOR THE MODEL SELECTION
models=[[100,0.01,0.1,0.01]
,[100,0.01,0.3,0.01]
,[100,0.01,0.6,0.01]
,[100,0.01,0.8,0.01]
,[100,0.01,0.1,0.001]
,[100,0.01,0.3,0.001]
,[100,0.01,0.6,0.001]
,[100,0.01,0.8,0.001]
,[100,0.01,0.1,0.0001]
,[100,0.01,0.3,0.0001]
,[100,0.01,0.6,0.0001]
,[100,0.01,0.8,0.0001]
,[50,0.01,0.1,0.01]
,[50,0.01,0.3,0.01]
,[50,0.01,0.6,0.01]
,[50,0.01,0.8,0.01]
,[50,0.01,0.1,0.001]
,[50,0.01,0.3,0.001]
,[50,0.01,0.6,0.001]
,[50,0.01,0.8,0.001]
,[50,0.01,0.1,0.0001]
,[50,0.01,0.3,0.0001]
,[50,0.01,0.6,0.0001]
,[50,0.01,0.8,0.0001]
,[50,0.03,0.1,0.01]
,[50,0.03,0.3,0.01]
,[50,0.03,0.1,0.001]
,[50,0.03,0.3,0.001]
,[50,0.03,0.1,0.0001]
,[50,0.03,0.3,0.0001]
,[25,0.01,0.1,0.01]
,[25,0.01,0.3,0.01]
,[25,0.01,0.6,0.01]
,[25,0.01,0.8,0.01]
,[25,0.01,0.1,0.001]
,[25,0.01,0.3,0.001]
,[25,0.01,0.6,0.001]
,[25,0.01,0.8,0.001]
,[25,0.01,0.1,0.0001]
,[25,0.01,0.3,0.0001]
,[25,0.01,0.6,0.0001]
,[25,0.01,0.8,0.0001]
,[25,0.03,0.1,0.01]
,[25,0.03,0.3,0.01]
,[25,0.03,0.6,0.01]
,[25,0.03,0.8,0.01]
,[25,0.03,0.1,0.001]
,[25,0.03,0.3,0.001]
,[25,0.03,0.6,0.001]
,[25,0.03,0.8,0.001]
,[25,0.03,0.1,0.0001]
,[25,0.03,0.3,0.0001]
,[25,0.03,0.6,0.0001]
,[25,0.03,0.8,0.0001]
,[25,0.05,0.1,0.01]
,[25,0.05,0.3,0.01]
,[25,0.05,0.6,0.01]
,[25,0.05,0.1,0.001]
,[25,0.05,0.3,0.001]
,[25,0.05,0.6,0.001]
,[25,0.05,0.1,0.0001]
,[25,0.05,0.3,0.0001]
,[25,0.05,0.6,0.0001]
,[10,0.01,0.1,0.01]
,[10,0.01,0.3,0.01]
,[10,0.01,0.6,0.01]
,[10,0.01,0.8,0.01]
,[10,0.01,0.1,0.001]
,[10,0.01,0.3,0.001]
,[10,0.01,0.6,0.001]
,[10,0.01,0.8,0.001]
,[10,0.01,0.1,0.0001]
,[10,0.01,0.3,0.0001]
,[10,0.01,0.6,0.0001]
,[10,0.01,0.8,0.0001]
,[10,0.03,0.1,0.01]
,[10,0.03,0.3,0.01]
,[10,0.03,0.6,0.01]
,[10,0.03,0.8,0.01]
,[10,0.03,0.1,0.001]
,[10,0.03,0.3,0.001]
,[10,0.03,0.6,0.001]
,[10,0.03,0.8,0.001]
,[10,0.03,0.1,0.0001]
,[10,0.03,0.3,0.0001]
,[10,0.03,0.6,0.0001]
,[10,0.03,0.8,0.0001]
,[10,0.05,0.1,0.01]
,[10,0.05,0.3,0.01]
,[10,0.05,0.6,0.01]
,[10,0.05,0.1,0.001]
,[10,0.05,0.3,0.001]
,[10,0.05,0.6,0.001]
,[10,0.05,0.1,0.0001]
,[10,0.05,0.3,0.0001]
,[10,0.05,0.6,0.0001]
,[10,0.05,0.8,0.0001]
,[10,0.08,0.1,0.01]
,[10,0.08,0.3,0.01]
,[10,0.08,0.6,0.01]
,[10,0.08,0.1,0.001]
,[10,0.08,0.3,0.001]
,[10,0.08,0.6,0.001]
,[10,0.08,0.1,0.0001]
,[10,0.08,0.3,0.0001]
,[10,0.08,0.6,0.0001]
,[10,0.08,0.8,0.0001]
]

## MEE CALCULATOR
def accuracy_regression(output,y):
	y=y.transpose()
	temp=0.0
	for i in range(max(y.shape)):
		temp+=LA.norm(output[i]-y[i])
	return temp/max(y.shape)

def KfoldVALIDATION(X_train,y_train):
	k_fold = KFold(n_splits=5,shuffle=False)
	fold=[]
	names=["HU","LR","MO","LB"]
	for train, val in k_fold.split(X_train[0]):
		X=X_train.transpose()
		X_t=X[train]
		X_t=X_t.transpose()
		X_v=X[val]
		X_v=X_v.transpose()
		y=y_train.transpose()
		y_t=y[train]
		y_t=y_t.transpose()
		y_v=y[val]
		y_v=y_v.transpose()
		temp=[]
		for h in models:
			m=Model(X_t,y_t,X_v,y_v,[],[])
			m.ANNModel(hidden_unit=h[0],outputsize=2,
					learning_rate = h[1],
					momentum = h[2],
					lamb=h[3],
					activations="regression",
					loss="MSE")
			m.train(8000)
			for i in range(len(h)):
				print('%s: %f'%(names[i],h[i]))
			print('Training   MEE:%f'%m.accuracy_train)
			print('Validation MEE:%f'%m.accuracy_val)
			print('\n')
			temp.append([h,m.accuracy_train,m.cost_train,m.accuracy_val,m.cost_val])
		fold.append(temp)
	pickle.dump( fold, open( "foldsTESTmodel10.p", "wb" ) )

def KerasANN(X_train,y_train):
	k_fold = KFold(n_splits=5,shuffle=False)
	fold=[]
	names=["HU","LR","MO","LB"]
	for train, val in k_fold.split(X_train[0]):
		X=X_train.transpose()
		X_t=X[train]
		X_t=X_t.transpose()
		X_v=X[val]
		X_v=X_v.transpose()
		y=y_train.transpose()
		y_t=y[train]
		y_t=y_t.transpose()
		y_v=y[val]
		y_v=y_v.transpose()
		cvscores = []
		#print(X_train.shape[0])
		for h in models:

			### KERAS MODEL
			model = Sequential()
			model.add(Dense(h[0], input_dim=10,W_regularizer=l2(h[3]), init='normal', activation='sigmoid'))
			model.add(Dense(2, init='normal', W_regularizer=l2(h[3]),activation='linear'))
			# Compile model
			sgd = SGD(lr=h[1], momentum=h[2])
			model.compile(loss='mean_squared_error', optimizer=sgd,metrics=['mean_squared_error'])
			#print(X_t.shape,y_t.shape)
			#print(model.summary())
			for i in range(5000):
				model.train_on_batch(X_t.transpose(),y_t.transpose())
			scores = model.evaluate(X_v.transpose(),y_v.transpose(), verbose=0)
			prediction=model.predict_on_batch(X_v.transpose())
			MEE=accuracy_regression(prediction,y_v)
			print([h,scores,MEE])
			cvscores.append([h,scores,MEE])
		fold.append(cvscores)
	pickle.dump( fold, open( "kerasModel10.p", "wb" ) )




if __name__=='__main__':
	X_train=pickle.load( open( "dataset/X_train.p", "rb" ) )
	y_train=pickle.load( open( "dataset/y_train.p", "rb" ) )
	X_test=pickle.load( open( "dataset/X_test.p", "rb" ) )
	y_test=pickle.load( open( "dataset/y_test.p", "rb" ) )
	X_blind_test=pickle.load( open( "dataset/X_blindteset.p", "rb" ) )

	## 5 fold cross Validation for my model
	#KfoldVALIDATION(X_train,y_train)


	## 5 fold cross Validation for Keras Model
	#KerasANN(X_train,y_train)
	## Test keras model
	#m=Model(X_train,y_train,X_test,y_test,X_test,y_test)
	#m.KerasANN(hidden_unit=25,learning_rate =0.05,momentum = 0.6,lamb=0.0001)


	## Linear Model
	#m=Model(X_train,y_train,X_test,y_test,X_test,y_test)
	#m.LinerReg()



	m=Model(X_train,y_train,X_test,y_test,X_test,y_test)
	m.ANNModel(hidden_unit=25,	outputsize=2,learning_rate =0.05,momentum = 0.6,lamb=0.0001, activations="regression",	loss="MSE")
	m.train(11000)
	predictedvalue=m.predict(X_blind_test)
	print('# Andrea Madotto\n# samurai\n# LOC-OSM2 - AA1 2016 CUP v1\n# 27 Dec 2016\n')
	i=1
	for x_i,y_i in zip(X_blind_test.transpose(),predictedvalue.transpose()):
		print('%d,%6f,%6f'%(i,y_i[0],y_i[1]))
		i+=1
	#m.test(X_test,y_test)
	#m.plotLA()


	### to generate avg and std of our model
	'''
	temp=[]
	for i in range(10):
		m=Model(X_train,y_train,X_test,y_test,X_test,y_test)
		m.ANNModel(hidden_unit=25,	outputsize=2,learning_rate =0.05,momentum = 0.6,lamb=0.0001, activations="regression",	loss="MSE")
		m.train(5000)
		temp.append([m.accuracy_train,m.cost_train,m.accuracy_val,m.cost_val])


	a=np.array(temp)
	print(np.mean(a, axis=0))
	print(np.std(a,axis=0))

	#m.test(X_test,y_test)
	m.plotLA()
	'''


	### Hidden Unit vs Lambda
	'''
	#0.01,0.001,0.0001
	plot_arr=[]
	pair=[	[10,0.01],[25,0.01],[50,0.01],	[100,0.01],
		  	[10,0.01],[25,0.001],	[50,0.001],	[100,0.001],
		  	[10,0.01],[25,0.0001],[50,0.0001],[100,0.0001]]
	for p in pair:
		m=Model(X_train,y_train,X_train,y_train,[],[])
		m.ANNModel(hidden_unit=p[0],outputsize=2,learning_rate =0.03,momentum = 0.1,lamb=p[1], activations="regression",	loss="MSE")
		m.train(300)
		plot_arr.append(m.loss_training)
	pickle.dump( plot_arr, open( "Model/HUvsLAMBDA03.p", "wb" ) )
	pickle.dump( pair, open( "Model/pairHUvsLAMBDA03.p", "wb" ) )
	'''



	### Learning rate vs Momentum
	'''
	plot_arr=[]
	pair=[[0.1,0.1],[0.05,0.1],[0.03,0.1],[0.01,0.1],
		  [0.1,0.3],[0.05,0.3],[0.03,0.3],[0.01,0.3],
		  [0.1,0.6],[0.05,0.6],[0.03,0.6],[0.01,0.6],
		  [0.1,0.8],[0.05,0.8],[0.03,0.8],[0.01,0.8]]
	for p in pair:
		m=Model(X_train,y_train,X_train,y_train,[],[])
		m.ANNModel(hidden_unit=10,	outputsize=2,learning_rate = p[0],momentum = p[1],lamb=0.01, activations="regression",	loss="MSE")
		m.train(300)
		plot_arr.append(m.loss_training)
	pickle.dump( plot_arr, open( "Model/LRvsMOM10.p", "wb" ) )
	pickle.dump( pair, open( "Model/pairLRvsMOM10.p", "wb" ) )
	'''
