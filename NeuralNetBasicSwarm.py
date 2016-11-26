import numpy as np
from Utils import mean_squared_error as cost
from sklearn.utils import check_random_state
from scipy.special import expit as sigmoid


class BasicSwarm(object):
    def __init__(self, n_in, n_hidden, n_out, path,
                 min_weight=-3, max_weight=3, min_v=-2, max_v=2,
                 num_particles=10, random_state=None):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.path = path
        self.path_len = len(path)
        self.num_particles = num_particles
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_v = min_v
        self.max_v = max_v
        self.rng = check_random_state(random_state)
        self.cost = cost

        # particle weights and velocities
        self.p_weight = self.rng.uniform(self.min_weight,
                                         self.max_weight,
                                         (self.num_particles, self.path_len))

        self.p_v = self.rng.uniform(self.min_v, self.max_v,
                                    (self.num_particles, self.path_len))

        # particle bests
        self.p_best_scores = [np.inf for _ in xrange(self.num_particles)]
        self.p_best_weights = [np.zeros((self.path_len,)) for _ in xrange(self.num_particles)]

        # swarm best
        self.s_best_score = np.inf
        self.s_best_weight = np.zeros((self.path_len, ))

        # stats
        self.num_evals = 0

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

    def eval_pnn(self, W, path, p_index, X, y):
        # set up
        for i in xrange(len(W)):
            if path[i] is not None:
                W[i][path[i]] = self.p_weight[p_index][i]

        # evalute
        y_pred = self.forward(W, X)
        score = self.cost(y, y_pred)

        # update pbest
        if score < self.p_best_scores[p_index]:
            self.p_best_scores[p_index] = score
            self.p_best_weights[p_index] = np.copy(self.p_weight[p_index])

        self.num_evals += 1
        return score
   

    def get_num_evals(self):
        return self.num_evals