from __future__ import print_function
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD
import numpy as np
import itertools
import matplotlib.pyplot as plt

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
	X=[ np.array(e) for e in newX]
	y=[e[0] for e in data]
	X = np.array(X)
	y = [np.array(y)]

	#rng_state = np.random.get_state()
	#np.random.shuffle(X)
	#np.random.set_state(rng_state)
	#np.random.shuffle(y)
	X=X.transpose()
	print(len(X[0]))
	y=np.array(y)
	return X,y

if __name__=='__main__':
	X_train,y_train=data('dataset/monks-2.train')
	X_test,y_test=data('dataset/monks-2.test')
	assert X_train.shape[0]== X_test.shape[0]

	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=17, init='normal', activation='sigmoid'))
	model.add(Dense(1, init='normal', activation='sigmoid'))
	# Compile model
	sgd = SGD(lr=0.5, momentum=0.9, )
	model.compile(loss='mean_squared_error', optimizer=sgd,metrics=['accuracy'])
	print(X_train.shape, y_train.shape)

	history = model.fit(X_train.transpose(), y_train.transpose(), batch_size=169,nb_epoch=500, verbose=0)
	# list all data in history
	print(history.history.keys())
	 summarize history for accuracy
	plt.plot(history.history['acc'])
	#plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	#plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	#plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	#plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	#for i in range(700):
	#	model.train_on_batch(X_train.transpose(), y_train.transpose())
	#	loss_and_metrics = model.evaluate(X_train.transpose(), y_train.transpose(), batch_size=432)
	#	print (loss_and_metrics)
