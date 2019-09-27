import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

args = sys.argv

trial_file = args[1]
score_file = args[2]
save_file = args[3]

tgt_scores = []
nontgt_scores = []

with open(trial_file, 'r') as tf, open(score_file, 'r') as sf:
    tf_lines = tf.read().splitlines()
    trial_types = [x.split()[2] for x in tf_lines]
    sf_lines = sf.read().splitlines()
    scores = [float(x.split()[2]) for x in sf_lines]

    for ttype, score in zip(trial_types, scores):
        if ttype == "target":
            tgt_scores.append(score)
        else:
            nontgt_scores.append(score)

tgt_scores = sorted(tgt_scores)
nontgt_scores = sorted(nontgt_scores)

fit_tgt = stats.norm.pdf(tgt_scores, np.mean(tgt_scores), np.std(tgt_scores))
fit_nontgt = stats.norm.pdf(nontgt_scores, np.mean(nontgt_scores), np.std(nontgt_scores))

plt.plot(tgt_scores, fit_tgt, '-g')
plt.hist(tgt_scores, normed=True)

plt.plot(nontgt_scores, fit_nontgt, '-r')
plt.hist(nontgt_scores, normed=True)

plt.savefig(save_file, dpi=300)


