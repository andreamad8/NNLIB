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
models=[]

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
			for i in range(8000):
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

	## My Model
	m=Model(X_train,y_train,X_test,y_test,X_test,y_test)
	m.ANNModel(hidden_unit=25,	outputsize=2,learning_rate =0.05,momentum = 0.6,lamb=0.0001, activations="regression",	loss="MSE")
	m.train(11000)
	#m.test(X_test,y_test)
	#m.plotLA()


	pickle.dump( m, open( "Model/finalmodel.p", "wb" ) )
	predictedvalue=m.predict(X_blind_test)
	print('# Andrea Madotto\n# samurai\n# LOC-OSM2 - AA1 2016 CUP v1\n# 27 Dec 2016\n')
	i=1
	for x_i,y_i in zip(X_blind_test.transpose(),predictedvalue.transpose()):
		print('%d,%6f,%6f'%(i,y_i[0],y_i[1]))
		i+=1
	print(i)
