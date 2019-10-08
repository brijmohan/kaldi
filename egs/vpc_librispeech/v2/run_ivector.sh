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

# SRE16 trials
sre16_trials=data/test_clean_trial/trials
sre16_trials_tgl=data/test_clean_trial/trials_male
sre16_trials_yue=data/test_clean_trial/trials_female

tag="_vm1" # experiment tag
train_data=train_460${tag}
train_plda=train_plda_460${tag}
enroll_data=test_clean_enroll${tag}
trial_data=test_clean_trial${tag}

score_file=data/${trial_data}/scores
score_file_adapt=data/${trial_data}/scores_adapt
score_dist=data/${trial_data}/ivector_dist.png

stage=0
if [ $stage -le 0 ]; then

  # format the data as Kaldi data directories
  #for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
  for part in train-clean-100 train-clean-360; do
    # use underscore-separated names in data directories.
    local/data_prep_adv.sh $data/LibriSpeech${tag}/$part data/$(echo $part | sed s/-/_/g)${tag}
  done

  # Combine all training data into one
  utils/combine_data.sh data/${train_data} \
	  data/train_clean_100${tag} data/train_clean_360${tag}

  # Make enrollment and trial data
  python local/make_librispeech_eval.py ./proto $data/LibriSpeech/test-clean
  utils/utt2spk_to_spk2utt.pl data/test_clean_enroll/utt2spk > data/test_clean_enroll/spk2utt
  utils/utt2spk_to_spk2utt.pl data/test_clean_trial/utt2spk > data/test_clean_trial/spk2utt
fi
#exit 0

nj=56
if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in ${train_data} ${enroll_data} ${trial_data} ; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 29 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 29 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

if [ $stage -le 2 ]; then
  # Train the UBM.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
    --nj $nj --num-threads 8  --subsample 1 \
    data/${train_data} 2048 \
    exp/diag_ubm${tag}

  sid/train_full_ubm.sh --cmd "$train_cmd --mem 25G" \
    --nj $nj --remove-low-count-gaussians false --subsample 1 \
    data/${train_data} \
    exp/diag_ubm${tag} exp/full_ubm${tag}
fi

if [ $stage -le 3 ]; then
  # Train the i-vector extractor.
  sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 35G" \
    --ivector-dim 600 \
    --num-iters 5 \
    --stage 3 \
    exp/full_ubm${tag}/final.ubm data/${train_data} \
    exp/extractor${tag}
    #--stage 4 \
fi

# In this section, we augment the SRE data with reverberation,
# noise, music, and babble, and combined it with the clean SRE
# data.  The combined list will be used to train the PLDA model.
if [ $stage -le 4 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/${train_data}/utt2num_frames > data/${train_data}/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we dont add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    data/${train_data} data/${train_data}_reverb
  cp data/${train_data}/vad.scp data/${train_data}_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/${train_data}_reverb data/${train_data}_reverb.new
  rm -rf data/${train_data}_reverb
  mv data/${train_data}_reverb.new data/${train_data}_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh ${data}/musan data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/${train_data} data/${train_data}_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/${train_data} data/${train_data}_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/${train_data} data/${train_data}_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/${train_data}_aug data/${train_data}_reverb data/${train_data}_noise data/${train_data}_music data/${train_data}_babble

  # Take a random subset of the augmentations (128k is somewhat larger than twice
  # the size of the SWBD+SRE list)
  #utils/subset_data_dir.sh data/${train_data}_aug 565000 data/${train_data}_aug_565k
  # Taking a subset of 260k because original 460h corpus is around 130k
  utils/subset_data_dir.sh data/${train_data}_aug 260000 data/${train_data}_aug_260k
  utils/fix_data_dir.sh data/${train_data}_aug_260k

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $nj --cmd "$train_cmd" \
    data/${train_data}_aug_260k exp/make_mfcc $mfccdir

  # Combine the clean and augmented SWBD+SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/${train_data}_combined data/${train_data}_aug_260k data/${train_data}

  # Filter out the clean + augmented portion of the SRE list.  This will be used to
  # train the PLDA model later in the script.
  utils/copy_data_dir.sh data/${train_data}_combined data/${train_plda}
  utils/filter_scp.pl data/${train_data}/spk2utt data/${train_data}_combined/spk2utt | utils/spk2utt_to_utt2spk.pl > data/${train_plda}/utt2spk
  utils/fix_data_dir.sh data/${train_plda}
fi

if [ $stage -le 5 ]; then
  # Extract i-vectors for SRE data (includes Mixer 6). We'll use this for
  # things like LDA or PLDA.
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
    exp/extractor${tag} data/${train_plda} \
    exp/ivectors_${train_plda}

  # The SRE16 major is an unlabeled dataset consisting of Cantonese and
  # and Tagalog.  This is useful for things like centering, whitening and
  # score normalization.
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
    exp/extractor${tag} data/${train_data} \
    exp/ivectors_${train_data}

  # The SRE16 test data
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 29 \
    exp/extractor${tag} data/${trial_data} \
    exp/ivectors_${trial_data}

  # The SRE16 enroll data
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 29 \
    exp/extractor${tag} data/${enroll_data} \
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

if [ $stage -le 8 ]; then
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

