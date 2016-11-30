import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import label_binarize, LabelEncoder
from DynamicOSI import DynamicOSI
from sklearn import cross_validation as cv

if __name__ == '__main__':
    rng = np.random.RandomState(0)
    X = np.genfromtxt('glass.data', delimiter=',')[:,:-1]
    Y = np.genfromtxt('glass.data', delimiter=',', usecols=[-1], dtype='str')
    le = LabelEncoder()
    y = le.fit_transform(Y)

    X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.5, random_state=rng)

    clf = DynamicOSI(n_hidden=6, num_particles=20, num_swarms=20, window=25, random_state=rng,
                     verbose=True, validation_size=0.33, method='worst-first')
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    mse = mean_squared_error(label_binarize(y_test, clf.classes_), y_pred)
    acc = accuracy_score(y_test, y_pred.argmax(axis=1))
    evals = clf.get_num_evals()
    print
    print "results:"
    print mse, acc, evals