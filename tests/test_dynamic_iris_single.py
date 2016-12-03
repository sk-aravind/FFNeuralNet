import numpy as np
from sklearn import datasets
from sklearn.preprocessing import label_binarize
from NeuralNetDynamicOSI import DynamicOSI
from sklearn import cross_validation as cv
from sklearn.metrics import accuracy_score, mean_squared_error


if __name__ == '__main__':
    rng = np.random.RandomState()
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = cv.train_test_split(iris.data, iris.target, test_size=0.5, random_state=rng)
    clf = DynamicOSI(n_hidden=[3], num_particles=20, num_dynamic_swarms=10, window=30, random_state=rng,
                     verbose=True, validation_size=0.33, method='worst-first')
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    mse = mean_squared_error(label_binarize(y_test, clf.classes_), y_pred)
    acc = accuracy_score(y_test, y_pred.argmax(axis=1))
    evals = clf.get_num_evals()
    print
    print "results:"
    print mse, acc, evals
