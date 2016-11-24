import numpy as np

def sigmoid(z):
    return 1. / (1 + np.exp(-z))
    
def tanh(z):
    return np.tanh(z)

def softmax(X):
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]
    return X

def mean_squared_error(y, y_pred):
    return np.average(((y_pred - y) ** 2).mean(axis=1))
