##NNLIB
In this project, we implemented a learning simulator system, i.e. a Neural Network. 
To train such network we used the Back Propagation algorithm in combination with Momentum,
and we also included an L2 regularization term.The code has been realized using Python (2.7) 
with the support of the deep learning library, such as Theano [1]. 

Further, we benchmarked our implementation using MONK datasets, and a regression task proposed in the Machine 
Learning course of the MSc in Computer Science of Pisa University ([AA1 cup data](http://pages.di.unipi.it/micheli/DID/CUP-AA1/2016/data2016.html)).
Finally, we compare our implementation with several existing model, such as:  NN using Keras [2], 
a linear model and SVR (Support Vector Regression) both using scikit-learn [3].
  
## Basic requirements
The code is written using python 2.7, but it is actually 
running also with python 3.6 since we used `from __future__ import print_function`) at the beginning of each file.
You need also to have installed the followings libraries:
- theano 
- keras 
- scikit-learn
- matplotlib
- seaborn 

## Library interface and Basic Usage
To access to the basic functionality of this library, you need to import:
```python
from annlib.model import Model
```
then you can access to the basic features of learning model (creation, train and test) using the for example the 
following interface:
```python 
	m=Model(X_train,y_train,X_val,y_val,X_test,y_test)
	m.ANNModel(hidden_unit=25,	outputsize=2,learning_rate =0.05,
             momentum = 0.6,lamb=0.0001, activations="regression",	
             loss="MSE")
	m.train(11000)
	m.test()
```
Using the following instruction is possible to run a basic banchmarks of the MONKS datasets. 
```python
python MONK.py
```
Different datasets can be selected changing the path of the load function inside each file. 
In the `AA1CUP.py` we also implemented two function used for the K-cross fold validation.

##References
[1] Al-Rfou, Rami, et al. "Theano: A Python framework for fast computation of mathematical expressions." arXiv preprint arXiv:1605.02688 (2016).
[2] Chollet, François. "Keras (2015)." URL http://keras. io.
[3] Pedregosa, Fabian, et al. "Scikit-learn: Machine learning in Python." Journal of Machine Learning Research 12.Oct (2011): 2825-2830.
