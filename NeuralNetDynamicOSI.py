import numpy as np
from scipy.stats import linregress
from sklearn import cross_validation as cv
from collections import deque
from sklearn.preprocessing import label_binarize
from NeuralNetBasicOSI import BasicOSI


class DynamicOSI(BasicOSI):
    def __init__(self, n_hidden, random_state=None, num_particles=20, num_dynamic_swarms=5, method='random',
                 min_weight=-3, max_weight=3, min_v=-2, max_v=2, window=10, validation_size=0.25,
                 c1=1.49445, c2=1.49445, w=0.729, verbose=False):
        super(DynamicOSI, self).__init__(n_hidden, random_state=random_state, num_particles=num_particles,
                                         min_weight=min_weight, max_weight=max_weight, min_v=min_v, max_v=max_v,
                                         window=window, validation_size=validation_size,
                                         c1=c1, c2=c2, w=w, verbose=verbose)
        self.num_dynamic_swarms = num_dynamic_swarms
        self.method = method

    def select_swarms(self, j, scores):
        if self.num_dynamic_swarms < 0:
            indices = np.arange(len(self.paths))
        elif self.method == "random":
            indices = self.select_swarms_random()
        elif self.method == "round-robin":
            indices = self.select_swarms_round_robin(j)
        elif self.method == "iterative":
            indices = self.select_swarms_iterative(j)
        elif self.method == "worst-first":
            indices = self.select_swarms_worst_probability(scores)
        else:
            raise ValueError('unsupported method: ' + self.method)

        return indices

    def select_swarms_random(self):
        num_swarms = min(len(self.paths), self.num_dynamic_swarms)
        indices = self.random_state.randint(0, len(self.paths), num_swarms)
        return indices

    def select_swarms_round_robin(self, index):
        indices = np.mod(np.arange(self.num_dynamic_swarms) + index, len(self.paths))
        return indices

    def select_swarms_iterative(self, index):
        start = self.num_dynamic_swarms * index
        end = start + self.num_dynamic_swarms
        indices = np.mod(np.arange(start, end), len(self.paths))
        return indices

    def select_swarms_worst_probability(self, scores):
        nscores = scores / np.sum(scores)
        indices = self.random_state.choice(np.arange(len(self.paths)), self.num_dynamic_swarms, replace=False, p=nscores)
        return indices

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
        tmp = [1e3 - float(x * 1e3) / self.window for x in xrange(self.window)]
        window = deque(tmp, maxlen=(self.window * 5))
        self.num_evals = 0
        best_score = np.inf
        swarm_scores = np.ones(len(self.paths, ))

        if self.verbose:
            print "Fitting network {0}-{1}-{2} with {3} paths".format(self.n_in, self.n_hidden, self.n_out, len(self.swarms))

        while True:
            j += 1
            indices = self.select_swarms(j, swarm_scores)
            for s_index in indices:
                for p_index in xrange(self.num_particles):
                    self.num_evals += 1

                    # evaluate each swarm
                    score = self.swarms[s_index].evaluate(W, X_train, y_train, p_index)
                    print score 

                    # reconstruct gvn
                    Wn = self.reconstruct_gvn(W)

                    # update
                    self.swarms[s_index].update(self.w, self.c1, self.c2, p_index)

                    # evaluate gvn
                    y_pred = self.forward(Wn, X_valid)
                    score = self.cost(y_valid, y_pred)
                    if score < best_score:
                        W = Wn[:]
                        best_score = score
                        x3 = best_score/3
                    if score < swarm_scores[s_index]:
                        swarm_scores[s_index] = score

            # check termination
            window.append(best_score)
            r = linregress(range(self.window), list(window)[-self.window:])
            if self.verbose:
                print j, best_score , x3


            if r[0] >= 0 or best_score < 1e-3:
                self.W = W
                self.num_generations = j
                return self
