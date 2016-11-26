import numpy as np
from sklearn.preprocessing import label_binarize, MultiLabelBinarizer
from sklearn import cross_validation as cv
from sklearn.metrics import mean_squared_error, accuracy_score
from NeuralNetBasicOSI import BasicOSI

if __name__ == '__main__':
    rng = np.random.RandomState()
    X = np.genfromtxt('../data/4bit.data', delimiter=',')[:, :-1]
    y = np.genfromtxt('../data/4bit.data', delimiter=',', usecols=[-1])
    X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.33, random_state=rng)
    clf = BasicOSI(n_hidden=[4, 3, 2], num_particles=10, window=10,
                   random_state=rng, verbose=True, validation_size=0.33)
    clf.fit(X_train, y_train)
    mlb = MultiLabelBinarizer()
    y_pred = clf.predict_proba(X_test)
    mse = mean_squared_error(mlb.fit_transform(label_binarize(y_test, clf.classes_)), y_pred)
    acc = accuracy_score(y_test, y_pred.argmax(axis=1))
    evals = clf.get_num_evals()
    print
    print "results:"
    print mse, acc, evals
