import numpy as np

testname = '4bit'

if __name__ == "__main__":
    fname = 'dynamic_' + testname + '_cv_results.txt'
    methods = ['random', 'round-robin', 'iterative', 'worst-first']
    scores = [3, 4, 5]
    scores_print = {3 : 'mse', 4 : 'accuracy', 5 : 'evaluations'}

    f = open(fname)
    d = [s.strip() for s in f.readlines() if s.count(',') == 6]
    data = np.array([np.array(s.split(',')) for s in d])

    # subset
    window = np.unique(np.array(data[:,2], dtype=float))
    particles = np.unique(np.array(data[:,3], dtype=float))

    for s in scores:
        print scores_print[s]
        for m in methods:
            print m
            tmp = data[data[:,0] == m][:,1:]
            tmp = np.array(tmp, dtype=float)
            for w in window:
                for p in particles:
                    res = tmp[np.logical_and(tmp[:,2] == p, tmp[:,1] == w)][:,s]
                    print " ".join([str(t) for t in res])
            print
        print