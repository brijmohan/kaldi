'''
python tsne_spk.py original_data_dir xvector_feat_file
'''

from os.path import join
import sys
import kaldi_io
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

args = sys.argv

data_dir = args[1]
if len(args) > 2:
    feat_file = args[2]
else:
    feat_file = join(data_dir, 'feats.scp')

spk2utt = {}
spk2gender = {}
spk2featlen = {}

feats = {}

MAX_UTT_PER_SPK = 20
X = []

MAX_MALE = 5
MAX_FEMALE = 5

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

print('Reading feats...')
with open(feat_file) as f:
    for line in f.read().splitlines():
        sp = line.split()
        feats[sp[0]] = sp[1]

print('Accumulating stats...')
mc, fc = 0, 0
with open(join(data_dir, 'spk2utt')) as f:
    for line in f.read().splitlines():
        sp = line.split()
        utts = sp[1:MAX_UTT_PER_SPK+1]
        

        if sp[0].split('-')[2].startswith('female'):
            if fc < MAX_FEMALE:
                fc += 1
                spk2gender[sp[0]] = 'f'
            else:
                continue
        else:
            if mc < MAX_MALE:
                mc += 1
                spk2gender[sp[0]] = 'm'
            else:
                continue

        spk2utt[sp[0]] = utts

        spk_feats = []
        for u in utts:
            utt_feat = kaldi_io.read_vec_flt(feats[u])
            utt_feat = utt_feat[np.newaxis, :]
            #print(utt_feat.shape)
            spk_feats.append(utt_feat)
        spk_feats = np.array(spk_feats)
        spk_feats = spk_feats.squeeze()

        spk2featlen[sp[0]] = spk_feats.shape[0]
        print(spk_feats.shape)
        X.append(spk_feats)

nspk = len(spk2gender.keys())
print("Number of speakers", nspk)

labels = []
print("creating labels for silhouette score...")
for i, (spk, featlen) in enumerate(spk2featlen.items()):
    labels.extend([i]*featlen)

print('Computing TSNE...')
X = np.concatenate(X, axis=0)
mean_X = np.mean(X, axis=0)
std_X = np.std(X, axis=0)

X = (X - mean_X) / std_X

print("silhoutte X ", silhouette_score(X, labels))

Y = TSNE(perplexity=50).fit_transform(X)

print("silhoutte Y ", silhouette_score(Y, labels))

print(Y.shape)

print('Plotting TSNE...')

fig = plt.figure()
ax1 = fig.add_subplot(111)

cmap = get_cmap(nspk, name='tab10')
start, end = 0, None
for i, (spk, featlen) in enumerate(spk2featlen.items()):
    end = start + featlen
    if spk2gender[spk] == 'm':
        col = 'g'
        mark = 'o'
    else:
        col = 'b'
        mark = '^'
    ax1.scatter(Y[start:end, 0], Y[start:end, 1], c=cmap(i), marker=mark)
    start = end

plt.savefig(join(data_dir, 'tsne_uttxvector.png'), dpi=300)

