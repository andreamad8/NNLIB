# -*- coding: utf-8 -*- 
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

def KfoldVALIDATION(X_train,y_train):
	k_fold = KFold(n_splits=5,shuffle=False)
	fold=[]
	dists={
	'hidden_unit':   [10,25,50,100]
	,'learning_rate':[0.05,0.01,0.001]
	,'momentum':     [0.1,0.5,0.9]
	,'lamb':         [0.01,0.001,0.0001]
	,'MAX_EPOCHS':   [1000,3000]
	}
	names=["MAX_EPOCHS","hidden_unit","lamb","learning_rate","momentum",]
	
	for train, val in k_fold.split(X_train[0]):

		allNames = sorted(dists)
		combinations = it.product(*(dists[Name] for Name in allNames))
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
		for h in list(combinations):
			m=Model(X_t,y_t,X_v,y_v,[],[])
			m.ANNModel(hidden_unit=h[1],outputsize=2,
					learning_rate = h[3],
					momentum = h[4],
					lamb=h[2],
					activations="regression",
					loss="MSE")
			m.train(h[0])
			for i in range(len(h)):
				print('%s: %f'%(names[i],h[i]))
			print('Training   MEE:%f'%m.accuracy_train)
			print('Validation MEE:%f'%m.accuracy_val)
			print('\n')
			temp.append([h,m.accuracy_train,m.cost_train,m.accuracy_val,m.cost_val])
		fold.append(temp)
	pickle.dump( fold, open( "foldsTEST_norm.p", "wb" ) )

if __name__=='__main__':
	X_train=pickle.load( open( "dataset/X_train_scaled.p", "rb" ) )
	y_train=pickle.load( open( "dataset/y_train_scaled.p", "rb" ) )
	X_test=pickle.load( open( "dataset/X_test_scaled.p", "rb" ) )
	y_test=pickle.load( open( "dataset/y_test_scaled.p", "rb" ) )
	#KfoldVALIDATION(X_train.astype('float32'),y_train.astype('float32'))
	
	m=Model(X_train,y_train,X_test,y_test,X_test,y_test)
	#m.LinerReg() # MEE=0.14 MSE=3.11
	#m.SVR()
	#m.Kridge()

	m.ANNModel(hidden_unit=100,
			outputsize=2,
			learning_rate = 0.05,
			momentum = 0.5,
			lamb=0.0001,
			activations="regression",
			loss="MSE")
	m.train(3000)
	m.test(X_test,y_test)
	m.plotLA()


