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



'''
utilits to print out the best models in each folds
'''
#### thereare two files because I have runned the fold separatelly.
#fold = pickle.load( open( "ResultModelSelection/foldsTEST.p", "rb" ) )
#fold1 = pickle.load( open( "ResultModelSelection/foldsTESTmodel10.p", "rb" ) )
#fold= [fold[i]+fold1[i] for i in range(len(fold))]

fold = pickle.load( open( "ResultModelSelection/kerasModel.p", "rb" ) )
fold1 = pickle.load( open( "ResultModelSelection/kerasModel10.p", "rb" ) )
fold= [fold[i]+fold1[i] for i in range(len(fold))]
for f in fold:
	test=sorted(f,key=lambda tup: tup[2],reverse=False)
	#print('new fold')
	for t in test[:2]:
		temp=[]
		for i in range(0,len(t[0])):
			temp.append(t[0][i])
		temp.append(t[1][1])
		temp.append(t[2])
		print('%d,%2f,%1f,%5f,%4f,%4f'%(int(temp[0]),temp[1],temp[2],temp[3],temp[4],temp[5]))
