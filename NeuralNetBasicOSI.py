import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.utils import check_random_state, array2d
from Utils import mean_squared_error as cost
from itertools import tee, izip, product
from scipy.special import expit as sigmoid
from scipy.stats import linregress
from sklearn import cross_validation as cv
from collections import deque
from operator import attrgetter
from sklearn.preprocessing import label_binarize, MultiLabelBinarizer
from NeuralNetBasicSwarm import BasicSwarm


class BasicOSI(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden, random_state=None, num_particles=20,
                 min_weight=-3, max_weight=3, window=10, validation_size=0.25,
                 c1=1.49445, c2=1.49445, w=0.729, min_v=-2, max_v=2, verbose=False):
        self.n_hidden = n_hidden
        self.random_state = random_state
        self.num_particles = num_particles
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_v = min_v
        self.max_v = max_v
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.window = window
        self.validation_size = validation_size
        self.verbose = verbose
        self.mlb = MultiLabelBinarizer()
        self.cost = cost

    @staticmethod
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return izip(a, b)

    def construct_paths(self):
        layers = list()
        layers.append(self.n_in)
        layers.extend([x for x in self.n_hidden])
        layers.append(self.n_out)

        # construct network
        tmp = [range(x) for x in layers]
        tmp = list(product(*tmp))
        paths = [list(self.pairwise(i)) for i in tmp]

        # add bias
        for i in xrange(len(layers) - 1):
            tmp = [range(x) if j > 0 else [x] for j, x in enumerate(layers[i:])]
            tmp = list(product(*tmp))
            tmp = [[None] * i + list(self.pairwise(j)) for j in tmp]
            paths.extend(tmp)

        return paths
    
    def reconstruct_gvn(self, W):
        w = W[:]

        # argmin(s_i)
        for i in xrange(len(self.W_swarms)):
            for s in self.W_swarms[i]:
                best = min(s, key=attrgetter('s_best_score'))
                if best.s_best_score is not np.inf:
                    p = s[0].path[i]
                    w[i][p] = best.s_best_weight[i]

        return w

    def init_params(self, X, y):
        self.random_state = check_random_state(self.random_state)
        self.classes_, y = np.unique(y, return_inverse=True)
        X = array2d(X)
        _, self.n_in = X.shape
        self.n_out = np.unique(y).shape[0]

    def init_network(self):
        # construct the global network, +1 represents the bias layer
        W = list()
        for i in xrange(len(self.n_hidden)):
            if i == 0:
                W.append(self.random_state.uniform(self.min_weight, self.max_weight, (self.n_in + 1, self.n_hidden[i])))
            else:
                W.append(self.random_state.uniform(self.min_weight, self.max_weight, (self.n_hidden[i-1] + 1, self.n_hidden[i])))

        W.append(self.random_state.uniform(self.min_weight, self.max_weight, (self.n_hidden[-1] + 1, self.n_out)))

        # initialize the swarms
        self.swarms = [BasicSwarm(self.n_in, self.n_hidden, self.n_out, path,
                                  num_particles=self.num_particles, random_state=self.random_state,
                                  min_weight=self.min_weight, max_weight=self.max_weight,
                                  min_v=self.min_v, max_v=self.max_v)
                       for path in self.paths]
        return W

    def fit(self, X, y):
        self.init_params(X, y)
        self.paths = self.construct_paths()
        num = len(self.paths[0])
        swarm_paths = [sorted(list(set([s[i] for s in self.paths if s[i] is not None]))) for i in xrange(num)]
        W = self.init_network()
        self.W_swarms = [[[s for s in self.swarms if s.path[j] == i] for i in swarm_paths[j]] for j in xrange(num)]

        X_train, X_valid, y_train, y_valid = cv.train_test_split(X, y, test_size=self.validation_size,
                                                                 random_state=self.random_state)

        # binarize true values
        if len(self.classes_) > 2:
            y_train = label_binarize(y_train, self.classes_)
            y_valid = label_binarize(y_valid, self.classes_)
        else:
            y_train = self.mlb.fit_transform(label_binarize(y_train, self.classes_))
            y_valid = self.mlb.fit_transform(label_binarize(y_valid, self.classes_))

        j = 0
        tmp = [1e3 - float(x * 1e3)/self.window for x in xrange(self.window)]
        window = deque(tmp, maxlen=(self.window * 5))
        self.num_evals = 0
        best_score = np.inf

        if self.verbose:
            print "Fitting network {0}-{1}-{2} with {3} paths".format(self.n_in, self.n_hidden, self.n_out, len(self.swarms))

        while True:
            j += 1
            for s in self.swarms:
                for p_index in xrange(self.num_particles):
                    self.num_evals += 1

                    # evaluate each swarm
                    score = s.evaluate(W, X_train, y_train, p_index)

                    # reconstruct gvn
                    Wn = self.reconstruct_gvn(W)

                    # update
                    s.update(self.w, self.c1, self.c2, p_index)

                    # evaluate gvn
                    y_pred = self.forward(Wn, X_valid)
                    score = self.cost(y_valid, y_pred)
                    if score < best_score:
                        W = Wn[:]
                        best_score = score

            window.append(best_score)
            r = linregress(range(self.window), list(window)[-self.window:])
            if self.verbose:
                print j, best_score

            if r[0] >= 0 or best_score < 1e-3:
                self.W = W
                self.num_generations = j
                return self

    def forward(self, W_in, X):
        # construct network
        tmp = X
        for i in xrange(len(W_in)):
            if i == 0:
                W, W_b = W_in[i][:self.n_in], W_in[i][self.n_in:]
            else:
                W, W_b = W_in[i][:self.n_hidden[i-1]], W_in[i][self.n_hidden[i-1]:]
            tmp = sigmoid(np.dot(tmp, W) + W_b)

        return tmp
                
    def decision_function(self, X):
        return self.forward(self.W, X)

    def predict(self, X):
        scores = self.decision_function(X)
        results = np.argmax(scores, axis=1)
        return self.classes_[results]

    def predict_proba(self, X):
        return self.decision_function(X)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def get_num_evals(self):
        return np.sum([s.get_num_evals() for s in self.swarms]) + self.num_evals

    def get_num_generations(self):
        return self.num_generations
