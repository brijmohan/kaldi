#!/bin/bash

src_data=$1
pool_data=$2
xvec_out_dir=$3
plda_dir=$4

src_dataname=$(basename $src_data)
pool_dataname=$(basename $pool_data)
src_xvec_dir=${xvec_out_dir}/xvectors_${src_dataname}
pool_xvec_dir=${xvec_out_dir}/xvectors_${pool_dataname}
affinity_scores_dir=${src_xvec_dir}/spk_pool_scores
pseudo_xvecs_dir=${src_xvec_dir}/pseudo_spk_xvecs

# Iterate over all the source speakers and generate 
# affinity distribution over anonymization pool
src_spk2gender=${src_data}/spk2gender
pool_spk2gender=${pool_data}/spk2gender
cut -d\  -f 1 ${src_spk2gender} | while read s; do
  bash local/anon/compute_spk_pool_affinity.sh ${plda_dir} ${src_xvec_dir} ${pool_xvec_dir} \
	   "$s" "${affinity_scores_dir}/affinity_${s}"
done

# Filter the scores based on gender and then sort them based on affinity. 
# Select the xvectors of 100 farthest speakers and average them to get pseudospeaker.
python local/anon/gen_pseudo_xvecs.py ${src_data} ${pool_data} ${affinity_scores_dir} \ 
	${xvec_out_dir} ${pseudo_xvecs_dir}

