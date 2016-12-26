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
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns



def data(filename):
	X=list()
	y=list()
	with open(filename, 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			X.append([float(x) for x in row[1:-2]])
			y.append([float(x) for x in row[-2:]])


	index = math.ceil(len(X) * 0.7)
	X_train=X[:int(index)]
	y_train=y[:int(index)]
	X_test=X[int(index):]
	y_test=y[int(index):]
	X_train=np.array(X_train, dtype=theano.config.floatX)
	y_train=np.array(y_train, dtype=theano.config.floatX)
	X_test=np.array(X_test, dtype=theano.config.floatX)
	y_test=np.array(y_test, dtype=theano.config.floatX)
	print(np.mean(np.mean(X_train,axis=0)))
	print(np.mean(np.std(X_train,axis=0)))


	#### Scale feature between [0,1]
	##min_max_scaler = preprocessing.MinMaxScaler()
	##X_train = min_max_scaler.fit_transform(X_train)
	##X_test = min_max_scaler.fit_transform(X_test)
	#### mean 0 and std 1
	#X_train = preprocessing.scale(X_train)
	#X_test = preprocessing.scale(X_test)
	#### mean and normalization
	#std_scale = preprocessing.StandardScaler().fit(X_train)
	#X_train= std_scale.transform(X_train)
	#X_test= std_scale.transform(X_test)


	###PLOT FEATURES
	fig = plt.figure(figsize=(10.0,30.0))
	i=1
	for t in X_train.transpose():
		ax=fig.add_subplot(2,5,i)
		plt.title('Feature %d'%i ,fontsize=25.0)
		#sns.kdeplot(t, shade=True, color="r")
		sns.distplot(t,color="r")
		i+=1
	plt.show()

	'''
	#### SHUFFLE DATA
	rng_state = np.random.get_state()
	np.random.shuffle(X)
	np.random.set_state(rng_state)
	np.random.shuffle(y)


	X_train=X_train.transpose()
	X_test=X_test.transpose()
	y_train=y_train.transpose()
	y_test=y_test.transpose()
	print(X_train.shape,X_test.shape)
	print(y_train.shape,y_test.shape)

	return X_train,y_train,X_test,y_test
	'''
if __name__=='__main__':
	X_train,y_train,X_test,y_test=data('dataset/LOC-OSM2-TR.csv')
	'''
	pickle.dump( X_train, open( "dataset/X_train.p", "wb" ) )
	pickle.dump( y_train, open( "dataset/y_train.p", "wb" ) )
	pickle.dump( X_test, open( "dataset/X_test.p", "wb" ) )
	pickle.dump( y_test, open( "dataset/y_test.p", "wb" ) )
	'''
