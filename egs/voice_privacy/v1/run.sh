#!/bin/bash
# Script for first voice privacy challenge 2020
#
# First make sure that path.sh contains correct paths for
# pyTools written by NII, and a compiled netcdf binary

. path.sh
. cmd.sh

set -e

#===== begin config =======
nj=20
stage=14

librispeech_corpus=/home/bsrivast/asr_data/LibriSpeech

anoni_pool="libritts_train_other_500" # change this to the data you want to use for anonymization pool
data_src= # Data to be anonymized, must be in Kaldi format
data_src_netcdf=/home/bsrivast/asr_data/LibriTTS/am_nsf_data # change this to dir where VC features data will be stored

# Chain model for PPG extraction
ivec_extractor=exp/nnet3_cleaned/extractor # change this to the ivector extractor trained by chain models
ivec_data_dir=exp/nnet3_cleaned # change this to the directory where ivectors will stored for your data

tree_dir=exp/chain_cleaned/tree_sp # change this to tree dir of your chain model
model_dir=exp/chain_cleaned/tdnn_1d_sp # change this to your pretrained chain model
lang_dir=data/lang_chain # change this to the land dir of your chain model

ppg_dir=exp/nnet3_cleaned # change this to the dir where PPGs will be stored

# x-vector extraction
xvec_nnet_dir=exp/0007_voxceleb_v2_1a/exp/xvector_nnet_1a # change this to pretrained xvector model downloaded from Kaldi website
anon_xvec_out_dir=${xvec_nnet_dir}/anon

plda_dir=${xvec_nnet_dir}/xvectors_train

pseudo_xvec_rand_level=spk  # spk (all utterances will have same xvector) or utt (each utterance will have randomly selected xvector)
cross_gender="false"        # true, same gender xvectors will be selected; false, other gender xvectors

eval1_enroll=eval1_enroll
eval1_trial=eval1_trial
eval2_enroll=eval2_enroll
eval2_trial=eval2_trial

anon_data_suffix=_anon_${pseudo_xvec_rand_level}_${cross_gender}

#=========== end config ===========

# Download pretrained models
if [ $stage -le -1 ]; then
  echo "Downloading only Voxceleb pretrained model currently."
  local/download_pretrained.sh
fi


# Extract xvectors from anonymization pool
if [ $stage -le 6 ]; then
  echo "Stage 6: Extracting xvectors for anonymization pool."
  local/featex/01_extract_xvectors.sh --nj $nj data/${anoni_pool} ${xvec_nnet_dir} \
	  ${anon_xvec_out_dir}
fi

# Make evaluation data
if [ $stage -le 7 ]; then
  echo "Stage 7: Making evaluation data"
  python local/make_librispeech_eval.py ./proto/eval1 ${librispeech_corpus}/test-clean "" || exit 1;
  python local/make_librispeech_eval2.py proto/eval2 ${librispeech_corpus} "" || exit 1;

  # Sort and fix all data directories
  for name in $eval1_enroll $eval1_trial $eval2_enroll $eval2_trial; do
    echo "Sorting data: $name"
    for f in `ls data/${name}`; do
      mv data/${name}/$f data/${name}/${f}.u
      sort -u data/${name}/${f}.u > data/${name}/$f
      rm data/${name}/${f}.u
    done
    utils/utt2spk_to_spk2utt.pl data/${name}/utt2spk > data/${name}/spk2utt
    utils/fix_data_dir.sh data/${name}
    utils/validate_data_dir.sh --no-feats --no-text data/${name}
  done
fi

# Extract xvectors from data which has to be anonymized
if [ $stage -le 8 ]; then
  echo "Stage 8: Extracting xvectors for source data."
  for name in $eval1_enroll $eval1_trial $eval2_enroll $eval2_trial; do
    local/featex/01_extract_xvectors.sh --nj $nj data/${name} ${xvec_nnet_dir} \
	  ${anon_xvec_out_dir} || exit 1;
  done
fi

# Extract pitch for source data
if [ $stage -le 9 ]; then
  echo "Stage 9: Pitch extraction for source data."
  for name in $eval1_enroll $eval1_trial $eval2_enroll $eval2_trial; do
    local/featex/02_extract_pitch.sh --nj ${nj} data/${name} || exit 1;
  done
fi

# Extract PPGs for source data
if [ $stage -le 10 ]; then
  echo "Stage 10: PPG extraction for source data."
  for name in $eval1_enroll $eval1_trial $eval2_enroll $eval2_trial; do
    local/featex/extract_ppg.sh --nj $nj --stage 0 ${name} \
	  ${ivec_extractor} ${ivec_data_dir}/ivectors_${name} \
	  ${tree_dir} ${model_dir} ${lang_dir} ${ppg_dir}/ppg_${name} || exit 1;
  done
fi

# Generate pseudo-speakers for source data
if [ $stage -le 11 ]; then
  echo "Stage 11: Generating pseudo-speakers for source data."
  for name in $eval1_enroll $eval1_trial $eval2_enroll $eval2_trial; do
    local/anon/make_pseudospeaker.sh --rand-level ${pseudo_xvec_rand_level} \
      	  --cross-gender ${cross_gender} \
	  data/${name} data/${anoni_pool} ${anon_xvec_out_dir} \
	  ${plda_dir} || exit 1;
  done
fi

# Create netcdf data for voice conversion
if [ $stage -le 12 ]; then
  echo "Stage 12: Make netcdf data for VC."
  for name in $eval1_enroll $eval1_trial $eval2_enroll $eval2_trial; do
    local/anon/make_netcdf.sh --stage 0 data/${name} ${ppg_dir}/ppg_${name}/phone_post.scp \
	  ${anon_xvec_out_dir}/xvectors_${name}/pseudo_xvecs/pseudo_xvector.scp \
	  ${data_src_netcdf}/${name} || exit 1;
  done
fi

if [ $stage -le 13 ]; then
  echo "Stage 13: Extract melspec from acoustic model for each data."
  for name in $eval1_enroll $eval1_trial $eval2_enroll $eval2_trial; do
    local/vc/am/01_gen.sh ${data_src_netcdf}/${name} || exit 1;
  done
fi

if [ $stage -le 14 ]; then
  echo "Stage 14: Generate waveform from NSF model for each data."
  for name in $eval1_enroll $eval1_trial $eval2_enroll $eval2_trial; do
    local/vc/nsf/01_gen.sh ${data_src_netcdf}/${name} || exit 1;
  done
fi

if [ $stage -le 15 ]; then
  echo "Stage 15: Creating new data directories corresponding to anonymization."
  for name in $eval1_enroll $eval1_trial $eval2_enroll $eval2_trial; do
    wav_path=${data_src_netcdf}/${name}/nsf_output_wav
    new_data_dir=data/${name}${anon_data_suffix}
    utils/copy_data_dir.sh data/${name} ${new_data_dir}
    # Copy new spk2gender in case cross_gender vc has been done
    cp ${anon_xvec_out_dir}/xvectors_${name}/pseudo_xvecs/spk2gender ${new_data_dir}/
    awk -v p="$wav_path" '{print $1, p"/"$1".wav"}' data/${name}/wav.scp > ${new_data_dir}/wav.scp
  done
fi

if [ $stage -le 16 ]; then
  echo "Stage 16: Evaluate the dataset using speaker verification."
  echo "Exp 1: Eval 1, enroll - original, trial - anonymized"
  local/eval_libri.sh ${eval1_enroll} ${eval1_trial}${anon_data_suffix} || exit 1;
  echo "Exp 2: Eval 1, enroll - anonymized, trial - anonymized"
  local/eval_libri.sh ${eval1_enroll}${anon_data_suffix} ${eval1_trial}${anon_data_suffix} || exit 1;
  echo "Exp 3: Eval 2, enroll - original, trial - anonymized"
  local/eval_libri.sh ${eval2_enroll} ${eval2_trial}${anon_data_suffix} || exit 1;
  echo "Exp 4: Eval 2, enroll - anonymized, trial - anonymized"
  local/eval_libri.sh ${eval2_enroll}${anon_data_suffix} ${eval2_trial}${anon_data_suffix} || exit 1;
fi




