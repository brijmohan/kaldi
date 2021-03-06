#!/usr/bin/env bash
#
# Based mostly on the TED-LIUM and Switchboard recipe
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe and Yenda Trmal)
# Apache 2.0
#
# This is a subset of run.sh to only perform recognition experiments with evaluation data

# Begin configuration section.
decode_nj=20
stage=0
enhancement=beamformit # for a new enhancement method,
                       # change this variable and stage 4
# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh


set -e # exit on error

# chime5 main directory path
# please change the path accordingly
chime5_corpus=/export/corpora4/CHiME5
json_dir=${chime5_corpus}/transcriptions
audio_dir=${chime5_corpus}/audio

# training and test data
train_set=train_worn_simu_u400k
test_sets="eval_${enhancement}_dereverb_ref"

# This script also needs the phonetisaurus g2p, srilm, beamformit
./local/check_tools.sh || exit 1

if [ $stage -le 4 ]; then
  # Beamforming using reference arrays
  # enhanced WAV directory
  enhandir=enhan
  dereverb_dir=${PWD}/wav/wpe/
  for dset in eval; do
    for mictype in u01 u02 u03 u04 u05 u06; do
      local/run_wpe.sh --nj 4 --cmd "$train_cmd --mem 120G" \
			      ${audio_dir}/${dset} \
			      ${dereverb_dir}/${dset} \
			      ${mictype}
    done
  done
  for dset in dev eval; do
    for mictype in u01 u02 u03 u04 u05 u06; do
      local/run_beamformit.sh --cmd "$train_cmd" \
			      ${dereverb_dir}/${dset} \
			      ${enhandir}/${dset}_${enhancement}_${mictype} \
			      ${mictype}
    done
  done

  for dset in eval; do
    local/prepare_data.sh --mictype ref "$PWD/${enhandir}/${dset}_${enhancement}_u0*" \
			  ${json_dir}/${dset} data/${dset}_${enhancement}_dereverb_ref
  done
fi

if [ $stage -le 6 ]; then
  # fix speaker ID issue (thanks to Dr. Naoyuki Kanda)
  # add array ID to the speaker ID to avoid the use of other array information to meet regulations
  # Before this fix
  # $ head -n 2 data/eval_beamformit_ref_nosplit/utt2spk
  # P01_S01_U02_KITCHEN.ENH-0000192-0001278 P01
  # P01_S01_U02_KITCHEN.ENH-0001421-0001481 P01
  # After this fix
  # $ head -n 2 data/eval_beamformit_ref_nosplit_fix/utt2spk
  # P01_S01_U02_KITCHEN.ENH-0000192-0001278 P01_U02
  # P01_S01_U02_KITCHEN.ENH-0001421-0001481 P01_U02
  for dset in ${test_sets}; do
    utils/copy_data_dir.sh data/${dset} data/${dset}_nosplit
    mkdir -p data/${dset}_nosplit_fix
    cp data/${dset}_nosplit/{segments,text,wav.scp} data/${dset}_nosplit_fix/
    awk -F "_" '{print $0 "_" $3}' data/${dset}_nosplit/utt2spk > data/${dset}_nosplit_fix/utt2spk
    utils/utt2spk_to_spk2utt.pl data/${dset}_nosplit_fix/utt2spk > data/${dset}_nosplit_fix/spk2utt
  done

  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  for dset in ${test_sets}; do
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}_nosplit_fix data/${dset}
  done
fi

if [ $stage -le 7 ]; then
  # Now make MFCC features.
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  mfccdir=mfcc
  for x in ${test_sets}; do
    steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" \
		       data/$x exp/make_mfcc/$x $mfccdir
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
    utils/fix_data_dir.sh data/$x
  done
fi

nnet3_affix=_${train_set}_cleaned_rvb

lm_suffix=

if [ $stage -le 18 ]; then
  # First the options that are passed through to run_ivector_common.sh
  # (some of which are also used in this script directly).

  # The rest are configs specific to this script.  Most of the parameters
  # are just hardcoded at this level, in the commands below.
  affix=1a   # affix for the TDNN directory name
  tree_affix=
  tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
  dir=exp/chain${nnet3_affix}/tdnn${affix}_sp

  # training options
  # training chunk-options
  chunk_width=140,100,160
  # we don't need extra left/right context for TDNN systems.
  chunk_left_context=0
  chunk_right_context=0
  
  utils/mkgraph.sh \
      --self-loop-scale 1.0 data/lang${lm_suffix}/ \
      $tree_dir $tree_dir/graph${lm_suffix} || exit 1;

  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      local/nnet3/decode.sh --affix 2stage --pass2-decode-opts "--min-active 1000" \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --frames-per-chunk 150 --nj $decode_nj \
        --ivector-dir exp/nnet3${nnet3_affix} \
        --graph-affix ${lm_suffix} \
        data/${data} data/lang${lm_suffix} \
        $tree_dir/graph${lm_suffix} \
        exp/chain${nnet3_affix}/tdnn1b_sp 
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

if [ $stage -le 20 ]; then
  # final scoring to get the official challenge result
  # please specify both dev and eval set directories so that the search parameters
  # (insertion penalty and language model weight) will be tuned using the dev set
  local/score_for_submit.sh \
      --dev exp/chain${nnet3_affix}/tdnn1b_sp/decode${lm_suffix}_dev_${enhancement}_dereverb_ref_2stage \
      --eval exp/chain${nnet3_affix}/tdnn1b_sp/decode${lm_suffix}_eval_${enhancement}_dereverb_ref_2stage
fi
