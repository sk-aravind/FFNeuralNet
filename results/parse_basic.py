import numpy as np

test_name = '4bit'

if __name__ == "__main__":
    fname = 'basic_' + test_name + '_cv_results.txt'

    f = open(fname, 'r')
    lines = [line.strip() for line in f.readlines() if line.count(',') == 4]
    data = np.array([l.split(',') for l in lines], dtype=float)

    # subset
    window = np.unique(np.array(data[:,0], dtype=float))
    particles = np.unique(np.array(data[:,1], dtype=float))

    for score in [2, 3, 4]:
        for w in window:
            ps = [data[np.logical_and(data[:,0] == w, data[:,1] == p)][:,score] for p in particles]
            print " ".join([str(x) for x in np.array(ps).flatten()])
        print
