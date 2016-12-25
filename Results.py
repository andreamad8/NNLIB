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


dists={
'hidden_unit': [10,50,100]
,'learning_rate':[0.9,0.1,0.01,0.001]
,'momentum':[0.5,0.7,0.9]
,'lamb':[0.01,0.001,0.0001]
,'MAX_EPOCHS':[5,50,200]
}

Names = sorted(dists)
fold = pickle.load( open( "foldsTEST.p", "rb" ) )
for f in fold:
	test=sorted(f,key=lambda tup: tup[2],reverse=False)
	for t in test[:2]:
		for i in range(0,len(t[0])):
			print('%s: %f'%(Names[i],t[0][i]))
		print('Training   MEE:%f'%t[1])
		print('Validation MEE:%f'%t[2])
		print('\n')
	print('\n')





##############################
## FOLD1 
## 	'hidden_unit':   [10,50,100]
##	,'learning_rate':[0.9,0.1,0.01,0.001]
##	,'momentum':     [0.5,0.7,0.9]
##	,'lamb':         [0.01,0.001,0.0001]
##	,'MAX_EPOCHS':   [5,50,200]
################################