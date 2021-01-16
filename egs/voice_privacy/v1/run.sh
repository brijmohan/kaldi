#!/bin/bash
# Script for first voice privacy challenge 2020
#
# First make sure that path.sh contains correct paths for
# pyTools written by NII, and a compiled netcdf binary

. path.sh
. cmd.sh

set -e
export LC_ALL=C

#===== begin config =======
nj=20
stage=2

librispeech_corpus=/home/bsrivast/asr_data/LibriSpeech
libritts_corpus=/home/bsrivast/asr_data/LibriTTS/LibriTTS

anoni_pool="libritts_train_other_500" # change this to the data you want to use for anonymization pool
data_netcdf=/home/bsrivast/asr_data/LibriTTS/am_nsf_data # change this to dir where VC features data will be stored

# Chain model for PPG extraction
ppg_type="256" # 256 or 346, as a string
ppg_model=exp/asr_ppg_model_${ppg_type} # change this to the PPG extractor trained using chain models
ppg_dir=${ppg_model}/nnet3_cleaned # change this to the dir where PPGs will be stored

# x-vector extraction configs
#xvec_nnet_dir=exp/0007_voxceleb_v2_1a/exp/xvector_nnet_1a # change this to pretrained xvector model downloaded from Kaldi website
xvec_nnet_dir=exp/xvec_extractor/xvector_nnet_1a # change this to pretrained xvector model downloaded from Kaldi website
anon_xvec_out_dir=${xvec_nnet_dir}/anon

# ASV_eval configs
asv_eval_model=exp/asv_eval_model
plda_dir=${asv_eval_model}/xvect_train_clean_360
#asv_eval_model=exp/asv_eval_model_pitch
#plda_dir=${asv_eval_model}/xvectors_train
xvector_pitch=false

# ASR_eval model
asr_eval_model=exp/asr_eval_model

# Anonymization configs
pseudo_xvec_rand_level=spk  # spk (all utterances will have same xvector) or utt (each utterance will have randomly selected xvector)
cross_gender="false"        # true, same gender xvectors will be selected; false, other gender xvectors
distance="plda"             # cosine or plda
proximity="farthest"         # nearest or farthest, should the farthest or nearest speakers must be selected during anonymization

eval1_enroll=eval1_enroll
eval1_trial=eval1_trial
eval2_enroll=eval2_enroll
eval2_trial=eval2_trial

anon_data_suffix="_ppg_${ppg_type}_level_${pseudo_xvec_rand_level}_crossgender_${cross_gender}_distance_${distance}_proximity_${proximity}"
score_dist_dir=exp/score_dist

#=========== end config ===========

# Download pretrained models
if [ $stage -le -1 ]; then
  printf "${GREEN}\nDownloading only Voxceleb pretrained model currently.${NC}\n"
  local/download_pretrained.sh
fi

# Extract xvectors from anonymization pool
if [ $stage -le 0 ]; then
  #local/data_prep_libritts.sh ${libritts_corpus}/train-other-500 data/${anoni_pool}
  #utils/fix_data_dir.sh data/${anoni_pool}
  #utils/vaidate_data_dir.sh data/${anoni_pool}

  printf "${GREEN}\nStage 0: Extracting xvectors for anonymization pool.${NC}\n"
  local/featex/01_extract_xvectors.sh --nj 40 data/${anoni_pool} ${xvec_nnet_dir} \
	  ${anon_xvec_out_dir}
fi

# Make evaluation data
if [ $stage -le 1 ]; then
  printf "${GREEN}\nStage 1: Making evaluation data${NC}\n"
  local/make_eval1.sh proto/eval1 ${librispeech_corpus} ${eval1_enroll} ${eval1_trial}
  local/make_eval2.sh proto/eval2 ${librispeech_corpus} ${eval2_enroll} ${eval2_trial}
fi

# Extract xvectors from data which has to be anonymized
if [ $stage -le 2 ]; then
  printf "${GREEN}\nStage 2: Anonymizing eval1 and eval2 data.${NC}\n"
  #for name in $eval1_enroll $eval1_trial $eval2_enroll $eval2_trial; do
  #for name in $eval1_enroll $eval1_trial; do
  : '
  for name in $eval2_enroll $eval2_trial; do
    local/anon/anonymize_data_dir.sh --nj $nj --anoni-pool ${anoni_pool} \
	 --data-netcdf ${data_netcdf} \
	 --ppg-model ${ppg_model} --ppg-dir ${ppg_dir} --ppg-type ${ppg_type} \
	 --xvec-nnet-dir ${xvec_nnet_dir} \
	 --anon-xvec-out-dir ${anon_xvec_out_dir} --plda-dir ${plda_dir} \
	 --pseudo-xvec-rand-level ${pseudo_xvec_rand_level} --distance ${distance} \
	 --proximity ${proximity} \
	 --cross-gender ${cross_gender} --anon-data-suffix ${anon_data_suffix} \
	 ${name} || exit 1;
  done
  '
    local/anon/anonymize_data_dir.sh --nj $nj --anoni-pool ${anoni_pool} \
	 --data-netcdf ${data_netcdf} \
	 --ppg-model ${ppg_model} --ppg-dir ${ppg_dir} --ppg-type ${ppg_type} \
	 --xvec-nnet-dir ${xvec_nnet_dir} \
	 --anon-xvec-out-dir ${anon_xvec_out_dir} --plda-dir ${plda_dir} \
	 --pseudo-xvec-rand-level utt --distance ${distance} \
	 --proximity ${proximity} \
	 --cross-gender ${cross_gender} --anon-data-suffix ${anon_data_suffix} \
	 ${eval2_enroll} || exit 1;
    local/anon/anonymize_data_dir.sh --nj $nj --anoni-pool ${anoni_pool} \
	 --data-netcdf ${data_netcdf} \
	 --ppg-model ${ppg_model} --ppg-dir ${ppg_dir} --ppg-type ${ppg_type} \
	 --xvec-nnet-dir ${xvec_nnet_dir} \
	 --anon-xvec-out-dir ${anon_xvec_out_dir} --plda-dir ${plda_dir} \
	 --pseudo-xvec-rand-level spk --distance ${distance} \
	 --proximity ${proximity} \
	 --cross-gender ${cross_gender} --anon-data-suffix ${anon_data_suffix} \
	 ${eval2_trial} || exit 1;
fi

if [ $stage -le 3 ]; then
  printf "${GREEN}\nStage 3: Evaluate the dataset using speaker verification.${NC}\n"
  #printf "${RED}**Exp 0.1 baseline: Eval 1, enroll - original, trial - original**${NC}\n"
  #baseline_dist=${score_dist_dir}/${eval1_trial}_baseline.png
  #local/asv_eval.sh --score-dist ${baseline_dist} \
  #	  --nnet-dir ${asv_eval_model} --plda-dir ${plda_dir} \
  #	  --use-pitch ${xvector_pitch} \
  #	  ${eval1_enroll} ${eval1_trial} || exit 1;
  : '
  printf "${RED}**Exp 0.2 baseline: Eval 2, enroll - original, trial - original**${NC}\n"
  baseline_dist=${score_dist_dir}/${eval2_trial}_baseline.png
  local/asv_eval.sh --score-dist ${baseline_dist} \
  	  --nnet-dir ${asv_eval_model} --plda-dir ${plda_dir} \
  	  --use-pitch ${xvector_pitch} \
  	  ${eval2_enroll} ${eval2_trial} || exit 1;
  '
: '
  oa_dist_plot=${score_dist_dir}/${eval1_trial}${anon_data_suffix}_oa.png
  aa_dist_plot=${score_dist_dir}/${eval1_trial}${anon_data_suffix}_aa.png
  printf "${RED}**Exp 1: Eval 1, enroll - original, trial - anonymized**${NC}\n"
  local/asv_eval.sh --score-dist ${oa_dist_plot} --nnet-dir ${asv_eval_model} \
  	  --use-pitch ${xvector_pitch} \
  	  --plda-dir ${plda_dir} ${eval1_enroll} ${eval1_trial}${anon_data_suffix} || exit 1;
  printf "${RED}**Exp 2: Eval 1, enroll - anonymized, trial - anonymized**${NC}\n"
  local/asv_eval.sh --score-dist ${aa_dist_plot} --nnet-dir ${asv_eval_model} \
  	  --use-pitch ${xvector_pitch} \
  	 --plda-dir ${plda_dir} ${eval1_enroll}${anon_data_suffix} ${eval1_trial}${anon_data_suffix} || exit 1;
  '

  oa_dist_plot=${score_dist_dir}/${eval2_trial}${anon_data_suffix}_oa.png
  aa_dist_plot=${score_dist_dir}/${eval2_trial}${anon_data_suffix}_aa.png
  printf "${RED}**Exp 3: Eval 2, enroll - original, trial - anonymized**${NC}\n"
  local/asv_eval.sh --score-dist ${oa_dist_plot} --nnet-dir ${asv_eval_model} \
  	  --use-pitch ${xvector_pitch} \
  	  --plda-dir ${plda_dir} ${eval2_enroll} ${eval2_trial}${anon_data_suffix} || exit 1;
  printf "${RED}**Exp 4: Eval 2, enroll - anonymized, trial - anonymized**${NC}\n"
  local/asv_eval.sh --score-dist ${aa_dist_plot} --nnet-dir ${asv_eval_model} \
  	  --use-pitch ${xvector_pitch} \
  	 --plda-dir ${plda_dir} ${eval2_enroll}${anon_data_suffix} ${eval2_trial}${anon_data_suffix} || exit 1;
fi

exit 0;

if [ $stage -le 4 ]; then
  printf "${GREEN}\nStage 4: Anonymizing adaptation data to adapt speaker verification PLDA.${NC}\n"
  local/data_prep_adv.sh ${librispeech_corpus}/dev-other data/dev_other
  
  local/anon/anonymize_data_dir.sh --nj $nj --anoni-pool ${anoni_pool} \
	 --data-netcdf ${data_netcdf} \
	 --ppg-model ${ppg_model} --ppg-dir ${ppg_dir} --ppg-type ${ppg_type} \
	 --xvec-nnet-dir ${xvec_nnet_dir} \
	 --anon-xvec-out-dir ${anon_xvec_out_dir} --plda-dir ${plda_dir} \
	 --pseudo-xvec-rand-level ${pseudo_xvec_rand_level} --distance ${distance} \
	 --proximity ${proximity} \
	 --cross-gender ${cross_gender} --anon-data-suffix ${anon_data_suffix} \
	 dev_other || exit 1;
  
  adapt_data=dev_other${anon_data_suffix}
  local/featex/01_extract_xvectors.sh --nj $nj data/${adapt_data} ${xvec_nnet_dir} \
	  ${anon_xvec_out_dir}

  printf "${RED}Adapting the VoxCeleb model to ${adapt_data}...${NC}\n"
  $train_cmd ${anon_xvec_out_dir}/xvectors_${adapt_data}/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    $plda_dir/plda \
    "ark:ivector-subtract-global-mean scp:${anon_xvec_out_dir}/xvectors_${adapt_data}/xvector.scp ark:- | transform-vec ${plda_dir}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${anon_xvec_out_dir}/xvectors_${adapt_data}/plda || exit 1;
fi

if [ $stage -le 5 ]; then
  printf "${GREEN}\nStage 5: Evaluate the dataset using ADAPTED speaker verification.${NC}\n"
  printf "${RED}\n**Exp 5: Eval 1, enroll - anonymized, trial - anonymized**${NC}\n"
  local/asv_eval.sh --nnet-dir ${xvec_nnet_dir} \
	--plda-dir ${anon_xvec_out_dir}/xvectors_${adapt_data} \
	${eval1_enroll}${anon_data_suffix} ${eval1_trial}${anon_data_suffix} || exit 1;
  printf "${RED}**Exp 6: Eval 2, enroll - anonymized, trial - anonymized**${NC}\n"
  local/asv_eval.sh --nnet-dir ${xvec_nnet_dir} \
	--plda-dir ${anon_xvec_out_dir}/xvectors_${adapt_data} \
	${eval2_enroll}${anon_data_suffix} ${eval2_trial}${anon_data_suffix} || exit 1;
fi

# Not anonymizing train-clean-360 here since it takes enormous amount of time and memory
if [ $stage -le 6 ] && false; then
  printf "${GREEN}\nStage 6: Anonymizing train data for Informed xvector model.${NC}\n"
  local/data_prep_adv.sh ${librispeech_corpus}/train-clean-360 data/train_clean_360
  
  local/anon/anonymize_data_dir.sh --nj $nj --stage 4 --anoni-pool ${anoni_pool} \
	 --data-netcdf ${data_netcdf} \
	 --ppg-model ${ppg_model} --ppg-dir ${ppg_dir} --ppg-type ${ppg_type} \
	 --xvec-nnet-dir ${xvec_nnet_dir} \
	 --anon-xvec-out-dir ${anon_xvec_out_dir} --plda-dir ${plda_dir} \
	 --pseudo-xvec-rand-level ${pseudo_xvec_rand_level} --distance ${distance} \
	 --proximity ${proximity} \
	 --cross-gender ${cross_gender} --anon-data-suffix ${anon_data_suffix} \
	 train_clean_360 || exit 1;
  
  axvec_train_data=train_clean_360${anon_data_suffix}
fi

if [ $stage -le 7 ]; then
  printf "${GREEN}\nStage 7: Anonymizing test-clean data for intelligibility assessment.${NC}\n"
  local/data_prep_adv.sh ${librispeech_corpus}/test-clean data/test_clean
  local/fix_data_dir.sh data/test_clean
  
  local/anon/anonymize_data_dir.sh --nj $nj --anoni-pool ${anoni_pool} \
	 --data-netcdf ${data_netcdf} \
	 --ppg-model ${ppg_model} --ppg-dir ${ppg_dir} --ppg-type ${ppg_type} \
	 --xvec-nnet-dir ${xvec_nnet_dir} \
	 --anon-xvec-out-dir ${anon_xvec_out_dir} --plda-dir ${plda_dir} \
	 --pseudo-xvec-rand-level ${pseudo_xvec_rand_level} --distance ${distance} \
	 --proximity ${proximity} \
	 --cross-gender ${cross_gender} --anon-data-suffix ${anon_data_suffix} \
	 test_clean || exit 1;
fi

if [ $stage -le 8 ]; then
  asr_eval_data=test_clean${anon_data_suffix}
  printf "${GREEN}\nStage 8: Performing intelligibility assessment using ASR decoding on ${asr_eval_data}.${NC}\n"
  local/asr_eval.sh --nj $nj ${asr_eval_data} ${asr_eval_model}
fi
