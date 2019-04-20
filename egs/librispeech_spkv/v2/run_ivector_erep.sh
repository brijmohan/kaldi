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

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

# SRE16 trials
sre16_trials=data/erep_test_clean_trial_vad/trials
sre16_trials_tgl=data/erep_test_clean_trial_vad/trials_male
sre16_trials_yue=data/erep_test_clean_trial_vad/trials_female

stage=3
if [ $stage -le 0 ]; then
  # Path to some, but not all of the training corpora
  data_root=/export/corpora/LDC

  # Prepare telephone and microphone speech from Mixer6.
  local/make_mx6.sh $data_root/LDC2013S03 data/

  # Prepare SRE10 test and enroll. Includes microphone interview speech.
  # NOTE: This corpus is now available through the LDC as LDC2017S06.
  local/make_sre10.pl /export/corpora5/SRE/SRE2010/eval/ data/

  # Prepare SRE08 test and enroll. Includes some microphone speech.
  local/make_sre08.pl $data_root/LDC2011S08 $data_root/LDC2011S05 data/

  # This prepares the older NIST SREs from 2004-2006.
  local/make_sre.sh $data_root data/

  # Combine all SREs prior to 2016 and Mixer6 into one dataset
  utils/combine_data.sh data/sre \
    data/sre2004 data/sre2005_train \
    data/sre2005_test data/sre2006_train \
    data/sre2006_test_1 data/sre2006_test_2 \
    data/sre08 data/mx6 data/sre10
  utils/validate_data_dir.sh --no-text --no-feats data/sre
  utils/fix_data_dir.sh data/sre

  # Prepare SWBD corpora.
  local/make_swbd_cellular1.pl $data_root/LDC2001S13 \
    data/swbd_cellular1_train
  local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
    data/swbd_cellular2_train
  local/make_swbd2_phase1.pl $data_root/LDC98S75 \
    data/swbd2_phase1_train
  local/make_swbd2_phase2.pl /export/corpora5/LDC/LDC99S79 \
    data/swbd2_phase2_train
  local/make_swbd2_phase3.pl /export/corpora5/LDC/LDC2002S06 \
    data/swbd2_phase3_train

  # Combine all SWB corpora into one dataset.
  utils/combine_data.sh data/swbd \
    data/swbd_cellular1_train data/swbd_cellular2_train \
    data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

  # Prepare NIST SRE 2016 evaluation data.
  local/make_sre16_eval.pl /export/corpora5/SRE/R149_0_1 data

  # Prepare unlabeled Cantonese and Tagalog development data. This dataset
  # was distributed to SRE participants.
  local/make_sre16_unlabeled.pl /export/corpora5/SRE/LDC2016E46_SRE16_Call_My_Net_Training_Data data
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in test_clean_enroll test_clean_trial train_960_combined_no_sil train_plda_combined; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 16 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 16 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

<<c
# In this section, we augment the SRE data with reverberation,
# noise, music, and babble, and combined it with the clean SRE
# data.  The combined list will be used to train the PLDA model.
if [ $stage -le 2 ]; then
  utils/data/get_utt2num_frames.sh --nj 40 --cmd "$train_cmd" data/sre
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/sre/utt2num_frames > data/sre/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SRE list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    data/sre data/sre_reverb
  cp data/sre/vad.scp data/sre_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/sre_reverb data/sre_reverb.new
  rm -rf data/sre_reverb
  mv data/sre_reverb.new data/sre_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh /export/corpora/JHU/musan data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/sre data/sre_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/sre data/sre_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/sre data/sre_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/sre_aug data/sre_reverb data/sre_noise data/sre_music data/sre_babble

  # Take a random subset of the augmentations (64k is roughly the size of the SRE dataset)
  utils/subset_data_dir.sh data/sre_aug 64000 data/sre_aug_64k
  utils/fix_data_dir.sh data/sre_aug_64k

  # Make MFCCs for the augmented data.  Note that we want we should alreay have the vad.scp
  # from the clean version at this point, which is identical to the clean version!
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/sre_aug_64k exp/make_mfcc $mfccdir

  # Combine the clean and augmented SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/sre_combined data/sre_aug_64k data/sre
fi
c

train_data=erep_train_960_combined_no_sil_vad
train_plda=erep_train_plda_combined_vad
enroll_data=erep_test_clean_enroll_vad
trial_data=erep_test_clean_trial_vad

if [ $stage -le 3 ]; then
  # Train the UBM.
  : '
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
    --nj 50 --num-threads 50  --subsample 1 --no-vad true \
    data/${train_data} 2048 \
    exp/diag_ubm_erep
  '
  sid/train_full_ubm.sh --cmd "$train_cmd --mem 25G" \
    --nj 10 --remove-low-count-gaussians false --subsample 1 \
    --no-vad true --stage -1 \
    data/${train_data} \
    exp/diag_ubm_erep exp/full_ubm_erep
fi

if [ $stage -le 4 ]; then
  # Train the i-vector extractor.
  #utils/combine_data.sh data/swbd_sre data/swbd data/sre
  sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 35G" \
    --ivector-dim 600 \
    --num-iters 5 \
    --no-vad true \
    exp/full_ubm_erep/final.ubm data/${train_data} \
    exp/extractor_erep
fi

if [ $stage -le 5 ]; then
  # Extract i-vectors for SRE data (includes Mixer 6). We'll use this for
  # things like LDA or PLDA.
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 50 \
    --no-vad true \
    exp/extractor_erep data/${train_plda} \
    exp/ivectors_${train_plda}

  # The SRE16 major is an unlabeled dataset consisting of Cantonese and
  # and Tagalog.  This is useful for things like centering, whitening and
  # score normalization.
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 50 \
    --no-vad true \
    exp/extractor_erep data/${train_data} \
    exp/ivectors_${train_data}

  # The SRE16 test data
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 29 \
    --no-vad true \
    exp/extractor_erep data/${trial_data} \
    exp/ivectors_${trial_data}

  # The SRE16 enroll data
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 29 \
    --no-vad true \
    exp/extractor_erep data/${enroll_data} \
    exp/ivectors_${enroll_data}
fi

if [ $stage -le 6 ]; then
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

if [ $stage -le 7 ]; then
  # Get results using the out-of-domain PLDA model
  $train_cmd exp/scores_erep/log/sre16_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/ivectors_${enroll_data}/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_${train_plda}/plda - |" \
    "ark:ivector-mean ark:data/${enroll_data}/spk2utt scp:exp/ivectors_${enroll_data}/ivector.scp ark:- | ivector-subtract-global-mean exp/ivectors_${train_data}/mean.vec ark:- ark:- | transform-vec exp/ivectors_${train_plda}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_${train_data}/mean.vec scp:exp/ivectors_${trial_data}/ivector.scp ark:- | transform-vec exp/ivectors_${train_plda}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" exp/scores_erep/sre16_eval_scores || exit 1;

  utils/filter_scp.pl $sre16_trials_tgl exp/scores_erep/sre16_eval_scores > exp/scores_erep/sre16_eval_tgl_scores
  utils/filter_scp.pl $sre16_trials_yue exp/scores_erep/sre16_eval_scores > exp/scores_erep/sre16_eval_yue_scores
  pooled_eer=$(paste $sre16_trials exp/scores_erep/sre16_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  tgl_eer=$(paste $sre16_trials_tgl exp/scores_erep/sre16_eval_tgl_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  yue_eer=$(paste $sre16_trials_yue exp/scores_erep/sre16_eval_yue_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Out-of-Domain PLDA, EER: Pooled ${pooled_eer}%, Male ${tgl_eer}%, Female ${yue_eer}%"
  # EER: Pooled 13.65%, Tagalog 17.73%, Cantonese 9.612%
fi

if [ $stage -le 8 ]; then
  # Get results using an adapted PLDA model. In the future we'll replace
  # this (or add to this) with a clustering based approach to PLDA adaptation.
  $train_cmd exp/scores_erep/log/sre16_eval_scoring_adapt.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/ivectors_${enroll_data}/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_${train_data}/plda_adapt - |" \
    "ark:ivector-mean ark:data/${enroll_data}/spk2utt scp:exp/ivectors_${enroll_data}/ivector.scp ark:- | ivector-subtract-global-mean exp/ivectors_${train_data}/mean.vec ark:- ark:- | transform-vec exp/ivectors_${train_plda}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_${train_data}/mean.vec scp:exp/ivectors_${trial_data}/ivector.scp ark:- | transform-vec exp/ivectors_${train_plda}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" exp/scores_erep/sre16_eval_scores_adapt || exit 1;

  utils/filter_scp.pl $sre16_trials_tgl exp/scores_erep/sre16_eval_scores_adapt > exp/scores_erep/sre16_eval_tgl_scores_adapt
  utils/filter_scp.pl $sre16_trials_yue exp/scores_erep/sre16_eval_scores_adapt > exp/scores_erep/sre16_eval_yue_scores_adapt
  pooled_eer=$(paste $sre16_trials exp/scores_erep/sre16_eval_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  tgl_eer=$(paste $sre16_trials_tgl exp/scores_erep/sre16_eval_tgl_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  yue_eer=$(paste $sre16_trials_yue exp/scores_erep/sre16_eval_yue_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
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
