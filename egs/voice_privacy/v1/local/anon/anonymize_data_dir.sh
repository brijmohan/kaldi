#!/bin/bash
# Script for first voice privacy challenge 2020
#
# This script anonymizes a kaldi data directory and produces a new 
# directory with given suffix in the name

. path.sh
. cmd.sh

set -e

#===== begin config =======
nj=20
stage=0

anoni_pool="libritts_train_other_500" # change this to the data you want to use for anonymization pool
data_netcdf=/home/bsrivast/asr_data/LibriTTS/am_nsf_data # change this to dir where VC features data will be stored

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

anon_data_suffix=_anon_${pseudo_xvec_rand_level}_${cross_gender}

#=========== end config ===========

. utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: "
  echo "  $0 [options] <data-dir>"
  echo "Options"
  echo "   --nj=40     # Number of CPUs to use for feature extraction"
  exit 1;
fi

data_dir="$1" # Data to be anonymized, must be in Kaldi format

# Extract xvectors from data which has to be anonymized
if [ $stage -le 0 ]; then
  echo "Stage a.0: Extracting xvectors for ${data_dir}."
  local/featex/01_extract_xvectors.sh --nj $nj data/${data_dir} ${xvec_nnet_dir} \
	  ${anon_xvec_out_dir} || exit 1;
fi

# Generate pseudo-speakers for source data
if [ $stage -le 1 ]; then
  echo "Stage a.1: Generating pseudo-speakers for ${data_dir}."
  local/anon/make_pseudospeaker.sh --rand-level ${pseudo_xvec_rand_level} \
      	  --cross-gender ${cross_gender} \
	  data/${data_dir} data/${anoni_pool} ${anon_xvec_out_dir} \
	  ${plda_dir} || exit 1;
fi

# Extract pitch for source data
if [ $stage -le 2 ]; then
  echo "Stage a.2: Pitch extraction for ${data_dir}."
  local/featex/02_extract_pitch.sh --nj ${nj} data/${data_dir} || exit 1;
fi

# Extract PPGs for source data
if [ $stage -le 3 ]; then
  echo "Stage a.3: PPG extraction for ${data_dir}."
  local/featex/extract_ppg.sh --nj $nj --stage 0 ${data_dir} \
	  ${ivec_extractor} ${ivec_data_dir}/ivectors_${data_dir} \
	  ${tree_dir} ${model_dir} ${lang_dir} ${ppg_dir}/ppg_${data_dir} || exit 1;
fi

# Create netcdf data for voice conversion
if [ $stage -le 4 ]; then
  echo "Stage a.4: Make netcdf data for VC."
  local/anon/make_netcdf.sh --stage 0 data/${data_dir} ${ppg_dir}/ppg_${data_dir}/phone_post.scp \
	  ${anon_xvec_out_dir}/xvectors_${data_dir}/pseudo_xvecs/pseudo_xvector.scp \
	  ${data_netcdf}/${data_dir} || exit 1;
fi

if [ $stage -le 5 ]; then
  echo "Stage a.5: Extract melspec from acoustic model for ${data_dir}."
  local/vc/am/01_gen.sh ${data_netcdf}/${data_dir} || exit 1;
fi

if [ $stage -le 6 ]; then
  echo "Stage a.6: Generate waveform from NSF model for ${data_dir}."
  local/vc/nsf/01_gen.sh ${data_netcdf}/${data_dir} || exit 1;
fi

if [ $stage -le 7 ]; then
  echo "Stage a.7: Creating new data directories corresponding to anonymization."
  wav_path=${data_netcdf}/${data_dir}/nsf_output_wav
  new_data_dir=data/${data_dir}${anon_data_suffix}
  cp -r data/${data_dir} ${new_data_dir}
  # Copy new spk2gender in case cross_gender vc has been done
  cp ${anon_xvec_out_dir}/xvectors_${data_dir}/pseudo_xvecs/spk2gender ${new_data_dir}/
  awk -v p="$wav_path" '{print $1, "sox", p"/"$1".wav", "-t wav -r 16000 -b 16 - |"}' data/${data_dir}/wav.scp > ${new_data_dir}/wav.scp
fi
