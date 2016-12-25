from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
from ANNLIB.model import Model
from sklearn import preprocessing
import itertools
from sklearn.model_selection import KFold, cross_val_score
import csv
import math
import itertools as it
import pickle


# Data Set (v1): LOC-OSM2-TR
# Nov 2016
# AA1-2016 CUP: http://www.di.unipi.it/~micheli/DID/CUP-AA1/2016
# INFO: micheli@di.unipi.it
# (C) CIML group - Gallicchio, Micheli 2016
# 
# Format:
# Training set: id inputs target_x target_y (last 2 columns)
# Blind Test set: id inputs
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
,[25,0.05,0.6,0.0001]]

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
	pickle.dump( fold, open( "foldsTEST.p", "wb" ) )

if __name__=='__main__':
	X_train=pickle.load( open( "dataset/X_train.p", "rb" ) )
	y_train=pickle.load( open( "dataset/y_train.p", "rb" ) )
	X_test=pickle.load( open( "dataset/X_test.p", "rb" ) )
	y_test=pickle.load( open( "dataset/y_test.p", "rb" ) )
	#KfoldVALIDATION(X_train.astype('float32'),y_train.astype('float32'))
	
	m=Model(X_train,y_train,X_test,y_test,X_test,y_test)
	#m.LinerReg() # MEE=0.14 MSE=3.11
	#m.SVR()
	#m.Kridge()
	m=Model(X_train,y_train,X_test,y_test,X_test,y_test)
	m.ANNModel(hidden_unit=25,	outputsize=2,learning_rate =0.05,momentum = 0.6,lamb=0.001, activations="regression",	loss="MSE")
	m.train(8000)

	#m.test(X_test,y_test)
	m.plotLA()

	#0.01,0.001,0.0001
	#plot_arr=[]
	#pair=[[25,0.01],[50,0.01],[100,0.01],
	#	  [25,0.001],[50,0.001],[100,0.001],
	#	  [25,0.0001],[50,0.0001],[100,0.0001]]
	#for p in pair:
	#	m=Model(X_train,y_train,X_train,y_train,[],[])
	#	m.ANNModel(hidden_unit=p[0],outputsize=2,learning_rate =0.01,momentum = 0.1,lamb=p[1], activations="regression",	loss="MSE")
	#	m.train(300)
	#	plot_arr.append(m.loss_training)
	#pickle.dump( plot_arr, open( "Model/HUvsLAMBDA01.p", "wb" ) )
	#pickle.dump( pair, open( "Model/pairHUvsLAMBDA01.p", "wb" ) )

	




	#plot_arr=[]
	#pair=[[0.1,0.1],[0.05,0.1],[0.03,0.1],[0.01,0.1],
	#	  [0.1,0.3],[0.05,0.3],[0.03,0.3],[0.01,0.3],
	#	  [0.1,0.6],[0.05,0.6],[0.03,0.6],[0.01,0.6],
	#	  [0.1,0.8],[0.05,0.8],[0.03,0.8],[0.01,0.8]]
	#for p in pair:
	#	m=Model(X_train,y_train,X_train,y_train,[],[])
	#	m.ANNModel(hidden_unit=25,	outputsize=2,learning_rate = p[0],momentum = p[1],lamb=0.01, activations="regression",	loss="MSE")
	#	m.train(300)
	#	plot_arr.append(m.loss_training)
	#pickle.dump( plot_arr, open( "Model/LRvsMOM25.p", "wb" ) )
	#pickle.dump( pair, open( "Model/pairLRvsMOM25.p", "wb" ) )

	


