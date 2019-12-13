import sys
from os.path import basename, join
import operator

args = sys.argv

src_data = args[1]
pool_data = args[2]
affinity_scores_dir = args[3]
xvec_out_dir = args[4]
pseudo_xvecs_dir = args[5]

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
pool_xvec_dir = join(xvec_out_dir, 'xvectors_'+basename(pool_data),
                     'spk_xvector.scp')

for spk, gender in src_spk2gender.items():
    # Filter the affinity pool by gender
    affinity_pool = {}
    with open(join(affinity_scores_dir, 'affinity_'+spk)) as f:
        for line in f.read().splitlines():
            sp = line.split()
            pool_spk = sp[1]
            af_score = sp[2]
            if pool_spk2gender[pool_spk] == gender:
                affinity_pool[pool_spk] = af_score
    # Sort the filtered affinity pool by scores
    sorted_aff = sorted(affinity_pool.items(), key=operator.itemgetter(1))

