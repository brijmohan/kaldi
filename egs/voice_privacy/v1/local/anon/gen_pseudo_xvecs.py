import sys
from os.path import basename, join
import operator

import numpy as np

args = sys.argv

src_data = args[1]
pool_data = args[2]
affinity_scores_dir = args[3]
xvec_out_dir = args[4]
pseudo_xvecs_dir = args[5]
rand_level = args[6]
cross_gender = args[7] == "true"

gender_rev = {'m': 'f', 'f': 'm'}
src_spk2gender_file = join(src_data, 'spk2gender')
pool_spk2gender_file = join(pool_data, 'spk2gender')

src_spk2gender = {}
pool_spk2gender = {}
with open(src_spk2gender_file) as f:
    for line in f.read().splitlines():
        sp = line.split()
        src_spk2gender[sp[0]] = sp[1]
with open(pool_spk2gender_file) as f:
    for line in f.read().splitlines():
        sp = line.split()
        pool_spk2gender[sp[0]] = sp[1]

# Read pool xvectors
pool_xvec_file = join(xvec_out_dir, 'xvectors_'+basename(pool_data),
                     'spk_xvector.scp')
pool_xvectors = {}
with open(pool_xvec_file) as f:
    for key, xvec in kaldi_io.read_vec_flt_scp(f):
        #print key, mat.shape
        pool_xvectors[key] = xvec

pseudo_xvec_map = {}
pseudo_gender_map = {}
for spk, gender in src_spk2gender.items():
    # Filter the affinity pool by gender
    affinity_pool = {}
    # If we are doing cross-gender VC, reverse the gender else gender remains same
    if cross_gender:
        gender = gender_rev[gender]
    pseudo_gender_map[spk] = gender
    with open(join(affinity_scores_dir, 'affinity_'+spk)) as f:
        for line in f.read().splitlines():
            sp = line.split()
            pool_spk = sp[1]
            af_score = sp[2]
            if pool_spk2gender[pool_spk] == gender:
                affinity_pool[pool_spk] = af_score

    # Sort the filtered affinity pool by scores
    sorted_aff = sorted(affinity_pool.items(), key=operator.itemgetter(1))

    # Select 500 least affinity speakers and then randomly select 100 out of
    # them
    top500_spk = sorted_aff[:500]
    random100mask = np.random.random_integers(0, 499, 100)
    pseudo_spk_list = [x for i, x in enumerate(top500_spk) if i in
                       random100mask]
    pseudo_spk_matrix = np.zeros((100, 512), dtype='float')
    for i, spk_aff in enumerate(pseudo_spk_list):
        pseudo_spk_matrix[i, :] = pool_xvectors[spk_aff[0]]
    # Take mean of 100 randomly selected xvectors
    pseudo_xvec = np.mean(pseudo_spk_matrix, axis=1)
    # Assign it to the current speaker
    pseudo_spk_map[spk] = pseudo_xvec





