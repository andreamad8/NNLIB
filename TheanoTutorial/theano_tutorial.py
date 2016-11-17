import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np

###########################
### Basic stuff with theano, 
### variable and gradient
### http://outlace.com/Beginner-Tutorial-Theano/
###########################

x = T.dscalar()

fx = T.exp(T.sin(x**2))
print type(fx) #just to show you that fx is a theano variable type

f = theano.function(inputs=[x], outputs=fx)
print f(10)

fp = T.grad(fx, wrt=x)
fprime = theano.function([x], fp)
print fprime(15)


def layer(x, w):
	b = np.array([1], dtype=theano.config.floatX)
	new_x = T.concatenate([x, b])
	m = T.dot(w.T, new_x) #theta1: 3x3 * x: 3x1 = 3x1 ;;; theta2: 1x4 * 4x1
	h = nnet.sigmoid(m)
	return h



def grad_desc(cost, theta):
    alpha = 0.1 #learning rate
    return theta - (alpha * T.grad(cost, wrt=theta))

x = T.dvector()
y = T.dscalar()

theta1 = theano.shared(np.array(np.random.rand(3,3), dtype=theano.config.floatX)) # randomly initialize
theta2 = theano.shared(np.array(np.random.rand(4,1), dtype=theano.config.floatX))


hid1 = layer(x, theta1) 
hid2= layer(hid1, theta2)
out1 = T.sum(hid2)
fc = (out1 - y)**2

cost = theano.function(inputs=[x, y], outputs=fc, updates=[(theta1, grad_desc(fc, theta1)), (theta2, grad_desc(fc, theta2))]) # not really sure about this 
run_forward = theano.function(inputs=[x], outputs=out1)

inputs = np.array([[0,1],[1,0],[1,1],[0,0]]).reshape(4,2) #training data X
print inputs
exp_y = np.array([1, 1, 0, 0]) #training data Y
cur_cost = 0
for i in range(10000):
    for k in range(len(inputs)):
        cur_cost = cost(inputs[k], exp_y[k]) #call our Theano-compiled cost function, it will auto update weights
    if i % 500 == 0: #only print the cost every 500 epochs/iterations (to save space)
        print('Cost: %s' % (cur_cost,))

print(run_forward([0,1]))
print(run_forward([1,1]))
print(run_forward([1,0]))
print(run_forward([0,0]))	



###########################
### Advanced theano
### variable, shared , gradient, Jacobian
### http://colinraffel.com/talks/next2015theano.pdf
###########################




x_1 = T.scalar('x_1')
# Now, we can define another variable bar which is just x_1 squared.
y_1 = x_1**2
# It will also 
f = theano.function(inputs=[x_1], outputs=y_1)
print(f(3))


#### variable decaration
A = T.matrix('A')
x = T.vector('x')
b = T.vector('b')
y = T.dot(A, x) + b

z = T.sum(A**2) # element wise

b_default = np.array([0, 0], dtype=theano.config.floatX)

linear_mix = theano.function(inputs=[A, x, theano.In(b, value=b_default)], outputs= [y, z])
#y= Ax+b   z=\sum (A^2)
# Supplying values for A, x, and b
print(linear_mix(np.array([[1, 2, 3],
                           [4, 5, 6]], dtype=theano.config.floatX), #A
                  np.array([1, 2, 3], dtype=theano.config.floatX),  #x
                  np.array([4, 5], dtype=theano.config.floatX)))    #b

# Using the default value for b
print(linear_mix(np.array([[1, 2, 3],
                           [4, 5, 6]], dtype=theano.config.floatX), #A
                  np.array([1, 2, 3], dtype=theano.config.floatX))) #x


##### shared variable

shared_var = theano.shared(np.array([[1, 2], [3, 4]], dtype=theano.config.floatX))
#to set the value
shared_var.set_value(np.array([[3, 4], [2, 1]], dtype=theano.config.floatX))
print("SHARED MATRIX")
print(shared_var.get_value())
shared_squared = shared_var**2
function_1 = theano.function([], shared_squared)
print("FUNCTION SQUARE")
print(function_1())

print("\n")
print("\n")
print("\n")

## update shared variable 
#definition 
subtract = T.matrix('subtract')
# updates takes a dict where keys are shared variables and values are the new value the shared variable should take
# Here, updates will set shared_var = shared_var - subtract

function_2 = theano.function([subtract], shared_var, updates={shared_var: shared_var - subtract})
print("shared_var before subtracting [[1, 1], [1, 1]] using function_2:")
print(shared_var.get_value())
# Subtract [[1, 1], [1, 1]] from shared_var
function_2(np.array([[1, 1], [1, 1]], dtype=theano.config.floatX))
print("shared_var after calling function_2:")
print(shared_var.get_value())
# Note that this also changes the output of function_1, because shared_var is shared!
print("New output of function_1() (shared_var**2):")
print(function_1())



#### GRADIENT 
# We can compute the gradient of y_1 with respect to x_1 like so:
# GRADIENT OF a scalar 
y_prime= T.grad(y_1, x_1)
print(y_1.eval({x_1: 10}))
print(y_prime.eval({x_1: 10}))

# we can do the same with matrixies 
# Recall that y = Ax + b
# We can also compute a Jacobian like so:
y_J = theano.gradient.jacobian(y, x)
linear_mix_J = theano.function([A, x, b], y_J)
# Because it's a linear mix, we expect the output to always be A
print(linear_mix_J(np.array([[9, 8, 7], 
                             [4, 5, 6]], dtype=theano.config.floatX), #A
                    np.array([1, 2, 3], dtype=theano.config.floatX),  #x
                    np.array([4, 5], dtype=theano.config.floatX)))    #b
# We can also compute the Hessian with theano.gradient.hessian (skipping that here)


y_J = theano.gradient.jacobian(y, x)
linear_mix_J = theano.function([A, x, b], y_J)
# Because it's a linear mix, we expect the output to always be A
print(linear_mix_J(np.array([[9, 8, 7], 
                             [4, 5, 6]], dtype=theano.config.floatX), #A
                    np.array([1, 2, 3], dtype=theano.config.floatX),  #x
                    np.array([4, 5], dtype=theano.config.floatX)))    #b
# We can also compute the Hessian with theano.gradient.hessian (skipping that here)