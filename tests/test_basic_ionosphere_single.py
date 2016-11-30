import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn import cross_validation as cv
from sklearn.preprocessing import LabelEncoder, label_binarize, MultiLabelBinarizer
from NeuralNetBasicOSI import BasicOSI


if __name__ == '__main__':
    rng = np.random.RandomState(0)
    X = np.genfromtxt('../data/ionosphere.data', delimiter=',')[:,:-1]
    Y = np.genfromtxt('../data/ionosphere.data', delimiter=',', usecols=[-1], dtype='str')
    le = LabelEncoder()
    y = le.fit_transform(Y)

    X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.5, random_state=rng)
    clf = BasicOSI(n_hidden=[5], num_particles=20, window=20,
                   random_state=rng, verbose=True, validation_size=0.33)
    clf.fit(X_train, y_train)
    mlb = MultiLabelBinarizer()
    y_pred = clf.predict_proba(X_test)
    mse = mean_squared_error(mlb.fit_transform(label_binarize(y_test, clf.classes_)), y_pred)
    acc = accuracy_score(y_test, y_pred.argmax(axis=1))
    evals = clf.get_num_evals()
    gens = clf.get_num_generations()
    print
    print "results:"
    print mse, acc, evals, gens
