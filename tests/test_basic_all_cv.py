import numpy as np
from sklearn import datasets
from sklearn.preprocessing import label_binarize, LabelEncoder, MultiLabelBinarizer
from NeuralNetBasicOSI import BasicOSI
from sklearn import cross_validation as cv
from sklearn.metrics import mean_squared_error, accuracy_score


def xval(clf, x, y, train_index, test_index):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)
    if len(clf.classes_) > 2:
        mse = mean_squared_error(label_binarize(y_test, clf.classes_), y_pred)
    else:
        mlb = MultiLabelBinarizer()
        mse = mean_squared_error(mlb.fit_transform(label_binarize(y_test, clf.classes_)), y_pred)
    acc = accuracy_score(y_test, y_pred.argmax(axis=1))
    evals = clf.get_num_evals()
    return mse, acc, evals


def test_iris():
    f = open("../results/basic_iris_cv_results.txt", 'w')
    rng = np.random.RandomState()
    params = {'window': [5, 10, 15, 20, 25, 30],
              'num_particles': [5, 10, 15, 20]}

    iris = datasets.load_iris()

    for w in params['window']:
        for p in params['num_particles']:
            # do a 5x2 cross val
            sss = cv.StratifiedShuffleSplit(iris.target, n_iter=5, test_size=0.5, random_state=rng)
            mses, accs, evals = [], [], []
            for train_index, test_index in sss:
                mse, acc, ev = xval(BasicOSI(n_hidden=[3], num_particles=p, window=w, random_state=rng,
                                             validation_size=0.33, verbose=False),
                                    iris.data, iris.target, train_index, test_index)
                mses.append(mse)
                accs.append(acc)
                evals.append(ev)
                mse, acc, ev = xval(BasicOSI(n_hidden=[3], num_particles=p, window=w, random_state=rng,
                                             validation_size=0.33, verbose=False),
                                    iris.data, iris.target, test_index, train_index)
                mses.append(mse)
                accs.append(acc)
                evals.append(ev)
            print ",".join(map(str, [w, p, np.mean(mses), np.mean(accs), np.mean(evals)]))
            f.write("\n" + ",".join(map(str, [w, p, np.mean(mses), np.mean(accs), np.mean(evals)])))
            f.write("\n" + ",".join(map(str, mses)))
            f.write("\n" + ",".join(map(str, accs)))
            f.write("\n" + ",".join(map(str, evals)))
            f.flush()
    f.close()


def test_ionosphere():
    f = open("../results/basic_ionosphere_cv_results.txt", 'w')
    rng = np.random.RandomState()
    params = {'window': [5, 10, 15, 20, 25, 30],
              'num_particles': [5, 10, 15, 20]}

    X = np.genfromtxt('../data/ionosphere.data', delimiter=',')[:, :-1]
    Y = np.genfromtxt('../data/ionosphere.data', delimiter=',', usecols=[-1], dtype='str')
    le = LabelEncoder()
    y = le.fit_transform(Y)

    for w in params['window']:
        for p in params['num_particles']:
            # do a 5x2 cross val
            sss = cv.StratifiedShuffleSplit(y, n_iter=5, test_size=0.5, random_state=rng)
            mses, accs, evals = [], [], []
            for train_index, test_index in sss:
                mse, acc, ev = xval(BasicOSI(n_hidden=[5], num_particles=p, window=w, random_state=rng,
                                             validation_size=0.33, verbose=False),
                                    X, y, train_index, test_index)
                mses.append(mse)
                accs.append(acc)
                evals.append(ev)
                mse, acc, ev = xval(BasicOSI(n_hidden=[5], num_particles=p, window=w, random_state=rng,
                                             validation_size=0.33, verbose=False),
                                    X, y, test_index, train_index)
                mses.append(mse)
                accs.append(acc)
                evals.append(ev)
            print ",".join(map(str, [w, p, np.mean(mses), np.mean(accs), np.mean(evals)]))
            f.write("\n" + ",".join(map(str, [w, p, np.mean(mses), np.mean(accs), np.mean(evals)])))
            f.write("\n" + ",".join(map(str, mses)))
            f.write("\n" + ",".join(map(str, accs)))
            f.write("\n" + ",".join(map(str, evals)))
            f.flush()
    f.close()


def test_glass():
    f = open("../results/basic_glass_cv_results.txt", 'w')
    rng = np.random.RandomState()
    params = {'window': [5, 10, 15, 20, 25, 30],
              'num_particles': [5, 10, 15, 20]}

    X = np.genfromtxt('../data/glass.data', delimiter=',')[:, :-1]
    Y = np.genfromtxt('../data/glass.data', delimiter=',', usecols=[-1], dtype='str')
    le = LabelEncoder()
    y = le.fit_transform(Y)

    for w in params['window']:
        for p in params['num_particles']:
            # do a 5x2 cross val
            sss = cv.StratifiedShuffleSplit(y, n_iter=5, test_size=0.5, random_state=rng)
            mses, accs, evals = [], [], []
            for train_index, test_index in sss:
                mse, acc, ev = xval(BasicOSI(n_hidden=[6], num_particles=p, window=w, random_state=rng,
                                             validation_size=0.33, verbose=False),
                                    X, y, train_index, test_index)
                mses.append(mse)
                accs.append(acc)
                evals.append(ev)
                mse, acc, ev = xval(BasicOSI(n_hidden=[6], num_particles=p, window=w, random_state=rng,
                                             validation_size=0.33, verbose=False),
                                    X, y, test_index, train_index)
                mses.append(mse)
                accs.append(acc)
                evals.append(ev)
            print ",".join(map(str, [w, p, np.mean(mses), np.mean(accs), np.mean(evals)]))
            f.write("\n" + ",".join(map(str, [w, p, np.mean(mses), np.mean(accs), np.mean(evals)])))
            f.write("\n" + ",".join(map(str, mses)))
            f.write("\n" + ",".join(map(str, accs)))
            f.write("\n" + ",".join(map(str, evals)))
            f.flush()
    f.close()


def test_4bit():
    f = open("../results/basic_4bit_cv_results.txt", 'w')
    rng = np.random.RandomState()
    params = {'window': [5, 10, 15, 20, 25, 30],
              'num_particles': [5, 10, 15, 20]}

    X = np.genfromtxt('../data/4bit.data', delimiter=',')[:, :-1]
    y = np.genfromtxt('../data/4bit.data', delimiter=',', usecols=[-1])

    for w in params['window']:
        for p in params['num_particles']:
            # do a 5x2 cross val
            sss = cv.StratifiedShuffleSplit(y, n_iter=5, test_size=0.5, random_state=rng)
            mses, accs, evals = [], [], []
            for train_index, test_index in sss:
                mse, acc, ev = xval(BasicOSI(n_hidden=[4, 3, 2], num_particles=p, window=w, random_state=rng,
                                             validation_size=0.33, verbose=False),
                                    X, y, train_index, test_index)
                mses.append(mse)
                accs.append(acc)
                evals.append(ev)
                mse, acc, ev = xval(BasicOSI(n_hidden=[4, 3, 2], num_particles=p, window=w, random_state=rng,
                                             validation_size=0.33, verbose=False),
                                    X, y, test_index, train_index)
                mses.append(mse)
                accs.append(acc)
                evals.append(ev)
            print ",".join(map(str, [w, p, np.mean(mses), np.mean(accs), np.mean(evals)]))
            f.write("\n" + ",".join(map(str, [w, p, np.mean(mses), np.mean(accs), np.mean(evals)])))
            f.write("\n" + ",".join(map(str, mses)))
            f.write("\n" + ",".join(map(str, accs)))
            f.write("\n" + ",".join(map(str, evals)))
            f.flush()
    f.close()

if __name__ == '__main__':
    test_iris()
    test_ionosphere()
    test_glass()
    test_4bit()
