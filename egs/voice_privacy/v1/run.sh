#!/bin/bash
# Script for first voice privacy challenge 2020
#
# First make sure that path.sh contains correct paths for
# pyTools written by NII, and a compiled netcdf binary

. path.sh
. cmd.sh

set -e

#===== begin config =======
nj=40
stage=0

librispeech_corpus=/home/bsrivast/asr_data/LibriSpeech

# Original data in ./data folder which will be splitted into train, dev and test based on speakers
am_nsf_train_data=libritts_train_clean_100 # change this to your actual data
anoni_pool="libritts_train_other_500" # change this to the data you want to use for anonymization pool
data_src= # Data to be anonymized, must be in Kaldi format
data_src_netcdf=/media/data/am_nsf_data/${data_src} # change this to dir where test data will be stored

# Chain model for PPG extraction
ivec_extractor=exp/nnet3_cleaned/extractor # change this to the ivector extractor trained by chain models
ivec_data_dir=exp/nnet3_cleaned # change this to the directory where ivectors will stored for your data

tree_dir=exp/chain_cleaned/tree_sp # change this to tree dir of your chain model
model_dir=exp/chain_cleaned/tdnn_1d_sp # change this to your pretrained chain model
lang_dir=data/lang_chain # change this to the land dir of your chain model

ppg_dir=exp/nnet3_cleaned # change this to the dir where PPGs will be stored

# Mel spectrogram config
am_nsf_melspec_dir=data/${train_data}_mspec
am_nsf_melspec_file=${melspec_dir}/feats.scp

# Split data
am_nsf_dev_spks=20
am_nsf_test_spks=20
am_nsf_split_dir=data/am_nsf

# x-vector extraction
am_nsf_train_split=${train_data}_train
am_nsf_dev_split=${train_data}_dev
am_nsf_test_split=${train_data}_test
am_nsf_split_data="${train_split} ${dev_split} ${test_split}"

xvec_nnet_dir=exp/0007_voxceleb_v2_1a/exp/xvector_nnet_1a # change this to pretrained xvector model downloaded from Kaldi website
am_nsf_xvec_out_dir=${xvec_nnet_dir}/am_nsf
anon_xvec_out_dir=${xvec_nnet_dir}/anon

plda_dir=${xvec_nnet_dir}/xvectors_train

# Output directories for netcdf data that will be used by AM & NSF training
am_nsf_train_out=/media/data/am_nsf_data/libritts/train_100 # change this to the dir where train, dev data and scp files will be stored
am_nsf_test_out=/media/data/am_nsf_data/libritts/test # change this to dir where test data will be stored

pseudo_xvec_rand_level=spk  # spk (all utterances will have same xvector) or utt (each utterance will have randomly selected xvector)
cross_gender="false"        # true, same gender xvectors will be selected; false, other gender xvectors

#=========== end config ===========

# Download pretrained models
if [ $stage -le -1 ]; then
  echo "Downloading only Voxceleb pretrained model currently."
  local/download_pretrained.sh
fi

# Extract PPG using chain model
if [ $stage -le 0 ]; then
  echo "Stage 0: PPG extraction."
  local/featex/extract_ppg.sh --nj $nj --stage 0 data/${am_nsf_train_data} \
	  ${ivec_extractor} ${ivec_data_dir}/ivectors_${am_nsf_train_data} \
	  ${tree_dir} ${model_dir} ${lang_dir} ${ppg_dir}/ppg_${am_nsf_train_data}
fi

# Extract 80 dimensional mel spectrograms
if [ $stage -le 1 ]; then
  echo "Stage 1: Mel spectrogram extraction."
  local/featex/extract_melspec.sh --nj $nj data/${am_nsf_train_data} ${am_nsf_melspec_dir}
fi

# Split the data into train, dev and test
if [ $stage -le 2 ]; then
  echo "Stage 2: Splitting the data into train, dev and test based on speakers."
  local/featex/00_make_am_nsf_data.sh --dev-spks ${am_nsf_dev_spks} --test-spks ${am_nsf_test_spks} \
	  data/${am_nsf_train_data} ${am_nsf_split_dir}
fi

# Extract xvectors from each split of data
if [ $stage -le 3 ]; then
  echo "Stage 3: x-vector extraction."
  for sdata in ${am_nsf_split_data}; do
    local/featex/01_extract_xvectors.sh --nj $nj ${am_nsf_split_dir}/${sdata} ${xvec_nnet_dir} \
	  ${am_nsf_xvec_out_dir}
  done
fi

# Extract pitch from each split of data
if [ $stage -le 4 ]; then
  echo "Stage 4: Pitch extraction."
  for sdata in ${am_nsf_split_data}; do
    local/featex/02_extract_pitch.sh --nj ${am_nsf_dev_spks} ${am_nsf_split_dir}/${sdata}
  done
fi

# Create NetCDF data from each split
if [ $stage -le 5 ]; then
  echo "Stage 5: Making netcdf data for AM & NSF training."
  local/featex/03_make_am_nsf_netcdf_data.sh ${am_nsf_train_split} ${am_nsf_dev_split} ${am_nsf_test_split} \
	  ${ppg_dir}/ppg_${am_nsf_train_data}/phone_post.scp ${am_nsf_melspec_file} \
	  ${am_nsf_xvec_out_dir} ${am_nsf_train_out} ${am_nsf_test_out}
fi

# Extract xvectors from anonymization pool
if [ $stage -le 6 ]; then
  echo "Stage 6: Extracting xvectors for anonymization pool."
  local/featex/01_extract_xvectors.sh --nj $nj data/${anoni_pool} ${xvec_nnet_dir} \
	  ${anon_xvec_out_dir}
fi

eval1_enroll=eval1_enroll
eval1_trial=eval1_trial
eval2_enroll=eval2_enroll
eval2_trial=eval2_trial

# Make evaluation data
if [ $stage -le 7 ]; then
  echo "Stage 7: Making evaluation data"
  python local/make_librispeech_eval.py ./proto/eval1 ${librispeech_corpus}/test-clean "" || exit 1;
  python local/make_librispeech_eval2.py proto/eval2 ${librispeech_corpus} "" || exit 1;

  for name in $eval1_enroll $eval1_trials $eval2_enroll $eval2_trial; do
    for f in `ls data/${name}`; do
      mv data/${name}/$f data/${name}/${f}.u
      sort -u data/${name}/${f}.u > data/${name}/$f
      rm data/${name}/${f}.u
    done
    utils/utt2spk_to_spk2utt.pl data/${name}/utt2spk > data/${name}/spk2utt
  done
fi

# Extract xvectors from data which has to be anonymized
if [ $stage -le 7 ]; then
  echo "Stage 7: Extracting xvectors for source data."
  local/featex/01_extract_xvectors.sh --nj $nj data/${data_anon} ${xvec_nnet_dir} \
	  ${anon_xvec_out_dir}
fi

# Extract pitch for source data
if [ $stage -le 8 ]; then
  echo "Stage 8: Pitch extraction for source data."
  local/featex/02_extract_pitch.sh --nj ${nj} data/${data_anon}
fi

# Extract PPGs for source data
if [ $stage -le 9 ]; then
  echo "Stage 9: PPG extraction for source data."
  local/featex/extract_ppg.sh --nj $nj --stage 0 data/${data_anon} \
	  ${ivec_extractor} ${ivec_data_dir}/ivectors_${data_anon} \
	  ${tree_dir} ${model_dir} ${lang_dir} ${ppg_dir}/ppg_${data_anon}
fi

# Generate pseudo-speakers for source data
if [ $stage -le 10 ]; then
  echo "Stage 10: Generating pseudo-speakers for source data."
  local/anon/make_pseudospeaker.sh --rand-level ${pseudo_xvec_rand_level} \
      --cross-gender ${cross_gender} \
	  data/${data_anon} data/${anoni_pool} ${anon_xvec_out_dir} \
	  ${plda_dir}
fi


# Create netcdf data for voice conversion
if [ $stage -le 11 ]; then
  echo "Stage 11: Make netcdf data for VC."
  local/anon/make_netcdf.sh --stage 0 data/${data_anon} ${ppg_dir}/ppg_${data_anon}/phone_post.scp \
	  ${anon_xvec_out_dir}/xvectors_${data_anon}/pseudo_xvecs/pseudo_xvector.scp \
	  ${data_src_netcdf}
fi
