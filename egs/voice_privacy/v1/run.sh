#!/bin/bash
# Script for first voice privacy challenge 2020
#
# First make sure that path.sh contains correct paths for
# pyTools written by NII, and a compiled netcdf binary

. path.sh
. cmd.sh

set -e

# begin config
nj=40
stage=0

# Original data in ./data folder which will be splitted into train, dev and test based on speakers
train_data=libritts_train_clean_100 # change this to your actual data

# Chain model for PPG extraction
ivec_extractor=exp/nnet3_cleaned/extractor # change this to the ivector extractor trained by chain models
ivec_data_dir=exp/nnet3_cleaned/ivector_${data} # change this to the directory where ivectors will stored for your data

tree_dir=exp/chain_cleaned/tree_sp # change this to tree dir of your chain model
model_dir=exp/chain_cleaned/tdnn_1d_sp # change this to your pretrained chain model
lang_dir=data/lang_chain # change this to the land dir of your chain model

ppg_dir=exp/nnet3_cleaned/ppg_wpd_fs1_${data} # change this to the dir where PPGs will be stored
ppg_file=${ppg_dir}/phone_post.scp

# Mel spectrogram config
melspec_dir=data/${train_data}_mspec
melspec_file=${melspec_dir}/feats.scp

# Split data
dev_spks=20
test_spks=20
split_dir=data/am_nsf

# x-vector extraction
train_split=${train_data}_train
dev_split=${train_data}_dev
test_split=${train_data}_test
split_data="${train_split} ${dev_split} ${test_split}"
xvec_nnet_dir=exp/0007_voxceleb_v2_1a/exp/xvector_nnet_1a # change this to pretrained xvector model downloaded from Kaldi website
xvec_out_dir=${xvec_nnet_dir}/am_nsf

# Output directories for netcdf data that will be used by AM & NSF training
train_out=/media/data/am_nsf_data/libritts/train_100 # change this to the dir where train, dev data and scp files will be stored
test_out=/media/data/am_nsf_data/libritts/test # change this to dir where test data will be stored
# end config

# Download pretrained models
if [ $stage -le -1 ]; then
  echo "Downloading only Voxceleb pretrained model currently."
  bash local/download_pretrained.sh
fi

# Extract PPG using chain model
if [ $stage -le 0 ]; then
  echo "Stage 0: PPG extraction."
  bash local/featex/extract_ppg.sh --nj $nj --stage 0 ${train_data} \
	  ${ivec_extractor} ${ivec_data_dir} ${tree_dir} ${model_dir} ${lang_dir} ${ppg_dir}
fi

# Extract 80 dimensional mel spectrograms
if [ $stage -le 1 ]; then
  echo "Stage 1: Mel spectrogram extraction."
  bash local/featex/extract_melspec.sh --nj $nj data/${train_data} ${melspec_dir}
fi

# Split the data into train, dev and test
if [ $stage -le 2 ]; then
  echo "Stage 2: Splitting the data into train, dev and test based on speakers."
  local/featex/00_make_am_nsf_data.sh --dev-spks ${dev_spks} --test-spks ${test_spks} \
	  data/${train_data}  ${split_dir}
fi

# Extract xvectors from each split of data
if [ $stage -le 3 ]; then
  echo "Stage 3: x-vector extraction."
  local/featex/01_extract_am_nsf_xvectors.sh --nj $nj ${split_dir} ${split_data} ${xvec_nnet_dir} \
	  ${xvec_out_dir}
fi

# Extract pitch from each split of data
if [ $stage -le 4 ]; then
  echo "Stage 4: Pitch extraction."
  local/featex/02_extract_am_nsf_pitch.sh --nj $nj ${split_dir} ${split_data}
fi

# Create NetCDF data from each split
if [ $stage -le 5 ]; then
  echo "Stage 5: Making netcdf data for AM & NSF training."
  local/featex/03_make_netcdf_data.sh ${train_split} ${dev_split} ${test_split} \
	  ${ppg_file} ${melspec_file} ${xvec_out_dir} ${train_out} ${test_out}
fi

