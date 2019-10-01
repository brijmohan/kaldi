'''
This script takes xvectors and classify gender for each point based on k-Nearest Neighbors

python local/xvector_gender_classification.py <xvector_folder>
'''

from os.path import join
import sys
import kaldi_io
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

args = sys.argv

feat_file = join(args[1], 'xvector.scp')

with open(feat_file) as f:
    lines = f.read().splitlines()
    npts = len(lines)
    test_x = kaldi_io.read_vec_flt(lines[0].split()[1])
    fdim = test_x.shape[0]
    
    X = np.zeros((npts, fdim))
    y = []
    for idx, line in enumerate(lines):
        sp = line.split()
        X[idx, :] = kaldi_io.read_vec_flt(sp[1])
        # male/female is present in uttname
        y.append(sp[0].split('-')[2].split('_')[0])

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y) 

print(neigh.score(X, y))

