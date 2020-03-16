from os.path import join

import kaldi_io
import matplotlib
matplotlib.use('Agg')

import numpy as np
import scipy.stats as stats
from scipy.stats import kurtosis, skew

import matplotlib.pyplot as plt
from numpy import linalg as LA

data_dir = 'data/kadv2.0460_test_clean_trial'
feats_scp = join(data_dir, 'feats.scp')
utt_norm_dist = join(data_dir, 'utt_norm_dist.png')
timestep_norm_dist = join(data_dir, 'timestep_norm_dist.png')

utt_norms = []
timestep_norms = []
for key, mat in kaldi_io.read_mat_scp(feats_scp):
    #print(key, mat.shape)
    # 1. Compute frobenius norm of the whole utterance
    utt_norms.append(LA.norm(mat))
    # 2. Compute frobenius norm of each timestep of the utterance
    timestep_norms.extend(LA.norm(mat, axis=1))

utt_norms = sorted(utt_norms)
timestep_norms = sorted(timestep_norms)

fit_utt = stats.norm.pdf(utt_norms, np.mean(utt_norms), np.std(utt_norms))
fit_timestep = stats.norm.pdf(timestep_norms, np.mean(timestep_norms), np.std(timestep_norms))

#print(skew(fit_utt), kurtosis(fit_utt))

plt.plot(utt_norms, fit_utt, '-r')
plt.hist(utt_norms, bins=50, normed=True, alpha=0.5)
m, v, s, k = round(float(np.mean(utt_norms)), 3), round(float(np.var(utt_norms)), 3), round(skew(fit_utt), 3), round(kurtosis(fit_utt), 3)
print(m, v, s, k)
plt.title("Mean {}, Var {}, Skew {}, Kurt {}".format(m, v, s, k))

plt.savefig(utt_norm_dist, dpi=300)

# Clear the figure
plt.clf()

plt.plot(timestep_norms, fit_timestep, '-r')
plt.hist(timestep_norms, bins=50, normed=True, alpha=0.5)
m, v, s, k = round(float(np.mean(timestep_norms)), 3), round(float(np.var(timestep_norms)), 3), round(skew(timestep_norms), 3), round(kurtosis(timestep_norms), 3)
print(m, v, s, k)
plt.title("Mean {}, Var {}, Skew {}, Kurt {}".format(m, v, s, k))

plt.savefig(timestep_norm_dist, dpi=300)

