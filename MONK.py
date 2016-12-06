from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
from ANNLIB.model import Model
from sklearn import preprocessing
import itertools
from sklearn.model_selection import KFold, cross_val_score

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
	y = [np.array(y, dtype=theano.config.floatX)]

	#rng_state = np.random.get_state()
	#np.random.shuffle(X)
	#np.random.set_state(rng_state)
	#np.random.shuffle(y)
	X=X.transpose()
	print(len(X[0]))
	y=np.array(y)
	return X,y

if __name__=='__main__':
	X_train,y_train=data('dataset/monks-1.train')
	X_test,y_test=data('dataset/monks-1.test')
	assert X_train.shape[0]== X_test.shape[0]
	

	m=Model(X_train,y_train,X_test,y_test,X_test,y_test)
	m.ANNModel(hidden_unit=4,
			outputsize=1,
			learning_rate = 0.9,
			momentum = 0.5,
			lamb=0.00,
			activations="sigmoid",
			loss="MSE")
	m.train(4000)
	#m.test()
	m.plotLA()