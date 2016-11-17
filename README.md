# NNLIB
My first neural network implementation
# Note
shuffle training data
X=X_train.transpose()
rng_state = np.random.get_state()
np.random.shuffle(X)
np.random.set_state(rng_state)
np.random.shuffle(y_train)
X_train=X.transpose()