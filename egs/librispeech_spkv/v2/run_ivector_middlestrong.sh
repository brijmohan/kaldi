#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2016 recipe using ivectors.
# In the future, we will add score-normalization and a more effective form of
# PLDA domain adaptation.

# This script runs ivectors over 460 hour subset of Librispeech

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

data=/home/bsrivastava/asr_data
data2=/home/bsrivastava/asr_data/LibriSpeech_dar_s2_2

# SRE16 trials
sre16_trials=data/test_clean_trial/trials
sre16_trials_tgl=data/test_clean_trial/trials_male
sre16_trials_yue=data/test_clean_trial/trials_female

tag="_dar_s2_16k" # vm1 = voicemask
tag2="_dar_s2_2"

train_data=train_460${tag}
train_plda=train_plda_460${tag}
enroll_data=test_clean_enroll${tag}
trial_data=test_clean_trial${tag2}
#train_data=train_460
#train_plda=train_plda_460
#enroll_data=test_clean_enroll
#trial_data=test_clean_trial

ivector_extractor=exp/extractor${tag}
#ivector_extractor=exp/extractor # Baseline model

score_file=data/${trial_data}/scores
score_file_adapt=data/${trial_data}/scores_adapt
score_dist=data/${trial_data}/ivector_dist.png

stage=2
if [ $stage -le 0 ]; then
: '
  # Sync VC transformed folders
  rsync -avzm --ignore-existing  $data/LibriSpeech/train-clean-360/* $data/LibriSpeech${tag}/train-clean-360/
  rsync -avzm --ignore-existing  $data/LibriSpeech/train-clean-100/* $data/LibriSpeech${tag}/train-clean-100/
  rsync -avzm --ignore-existing  $data/LibriSpeech/test-clean/* $data/LibriSpeech${tag}/test-clean/

  # format the data as Kaldi data directories
  #for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
  for part in train-clean-100 train-clean-360; do
    # use underscore-separated names in data directories.
    local/data_prep_adv.sh $data/LibriSpeech${tag}/$part data/$(echo $part | sed s/-/_/g)${tag}
  done

  # Combine all training data into one
  utils/combine_data.sh data/${train_data} \
	  data/train_clean_100${tag} data/train_clean_360${tag}
'
  # Make enrollment and trial data
  python local/make_librispeech_eval.py ./proto $data2/test-clean ${tag2}
  #utils/utt2spk_to_spk2utt.pl data/${enroll_data}/utt2spk > data/${enroll_data}/spk2utt
  utils/utt2spk_to_spk2utt.pl data/${trial_data}/utt2spk > data/${trial_data}/spk2utt
fi

nj=32
if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  #for name in dev_clean test_clean dev_other test_other train_960 test_clean_enroll test_clean_trial; do
  #for name in ${train_data} ${enroll_data} ${trial_data} ; do
  for name in ${trial_data} ; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 29 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 29 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

nj=29
if [ $stage -le 2 ]; then
  # The SRE16 test data
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 29 \
    --stage 1 \
    ${ivector_extractor} data/${trial_data} \
    exp/ivectors_${trial_data}
fi

if [ $stage -le 3 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd exp/ivectors_${train_data}/log/compute_mean.log \
    ivector-mean scp:exp/ivectors_${train_data}/ivector.scp \
    exp/ivectors_${train_data}/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd exp/ivectors_${train_plda}/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_${train_plda}/ivector.scp ark:- |" \
    ark:data/${train_plda}/utt2spk exp/ivectors_${train_plda}/transform.mat || exit 1;

  #  Train the PLDA model.
  $train_cmd exp/ivectors_${train_plda}/log/plda.log \
    ivector-compute-plda ark:data/${train_plda}/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_${train_plda}/ivector.scp ark:- | transform-vec exp/ivectors_${train_plda}/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/ivectors_${train_plda}/plda || exit 1;

  # Here we adapt the out-of-domain PLDA model to SRE16 major, a pile
  # of unlabeled in-domain data.  In the future, we will include a clustering
  # based approach for domain adaptation.
  $train_cmd exp/ivectors_${train_data}/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    exp/ivectors_${train_plda}/plda \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_${train_data}/ivector.scp ark:- | transform-vec exp/ivectors_${train_plda}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    exp/ivectors_${train_data}/plda_adapt || exit 1;
fi

if [ $stage -le 4 ]; then
  # Get results using the out-of-domain PLDA model
  $train_cmd exp/scores/log/sre16_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/ivectors_${enroll_data}/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_${train_plda}/plda - |" \
    "ark:ivector-mean ark:data/${enroll_data}/spk2utt scp:exp/ivectors_${enroll_data}/ivector.scp ark:- | ivector-subtract-global-mean exp/ivectors_${train_data}/mean.vec ark:- ark:- | transform-vec exp/ivectors_${train_plda}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_${train_data}/mean.vec scp:exp/ivectors_${trial_data}/ivector.scp ark:- | transform-vec exp/ivectors_${train_plda}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" ${score_file} || exit 1;

  utils/filter_scp.pl $sre16_trials_tgl ${score_file} > exp/scores/sre16_eval_tgl_scores
  utils/filter_scp.pl $sre16_trials_yue ${score_file} > exp/scores/sre16_eval_yue_scores
  pooled_eer=$(paste $sre16_trials $score_file | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  tgl_eer=$(paste $sre16_trials_tgl exp/scores/sre16_eval_tgl_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  yue_eer=$(paste $sre16_trials_yue exp/scores/sre16_eval_yue_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Out-of-Domain PLDA, EER: Pooled ${pooled_eer}%, Male ${tgl_eer}%, Female ${yue_eer}%"
  # EER: Pooled 13.65%, Tagalog 17.73%, Cantonese 9.612%
fi

if [ $stage -le 5 ]; then
  # Get results using an adapted PLDA model. In the future we'll replace
  # this (or add to this) with a clustering based approach to PLDA adaptation.
  $train_cmd exp/scores/log/sre16_eval_scoring_adapt.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/ivectors_${enroll_data}/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_${train_data}/plda_adapt - |" \
    "ark:ivector-mean ark:data/${enroll_data}/spk2utt scp:exp/ivectors_${enroll_data}/ivector.scp ark:- | ivector-subtract-global-mean exp/ivectors_${train_data}/mean.vec ark:- ark:- | transform-vec exp/ivectors_${train_plda}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_${train_data}/mean.vec scp:exp/ivectors_${trial_data}/ivector.scp ark:- | transform-vec exp/ivectors_${train_plda}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" ${score_file_adapt} || exit 1;

  utils/filter_scp.pl $sre16_trials_tgl ${score_file_adapt} > exp/scores/sre16_eval_tgl_scores_adapt
  utils/filter_scp.pl $sre16_trials_yue ${score_file_adapt} > exp/scores/sre16_eval_yue_scores_adapt
  pooled_eer=$(paste $sre16_trials ${score_file_adapt} | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  tgl_eer=$(paste $sre16_trials_tgl exp/scores/sre16_eval_tgl_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  yue_eer=$(paste $sre16_trials_yue exp/scores/sre16_eval_yue_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Adapted PLDA, EER: Pooled ${pooled_eer}%, Male ${tgl_eer}%, Female ${yue_eer}%"
  # EER: Pooled 12.98%, Tagalog 17.8%, Cantonese 8.35%
  #
  # Using the official SRE16 scoring software, we obtain the following equalized results:
  #
  # -- Pooled --
  # EER:         13.08
  # min_Cprimary: 0.72
  # act_Cprimary: 0.73

  # -- Cantonese --
  # EER:          8.23
  # min_Cprimary: 0.59
  # act_Cprimary: 0.59

  # -- Tagalog --
  # EER:         17.87
  # min_Cprimary: 0.84
  # act_Cprimary: 0.87
fi

if [ $stage -le 6 ]; then
    python local/plot_trial_score_dist.py $sre16_trials $score_file_adapt $score_dist 
    echo "Plot saved as:" $score_dist
fi
