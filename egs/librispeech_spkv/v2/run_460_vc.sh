#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2016 recipe using xvectors.
# It is closely based on "X-vectors: Robust DNN Embeddings for Speaker
# Recognition" by Snyder et al.  In the future, we will add score-normalization
# and a more effective form of PLDA domain adaptation.
#
# Pretrained models are available for this recipe.  See
# http://kaldi-asr.org/models.html and
# https://david-ryan-snyder.github.io/2017/10/04/model_sre16_v2.html
# for details.

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
#nnet_dir=exp/xvector_nnet_1a_460baseline_rm457 # CLEAN MODEL
#nnet_dir=exp/xvector_nnet_1a_460baseline_rm457_sg400k_s3 # VM1 MODEL
nnet_dir=exp/xvector_nnet_1a_460baseline_rm457_vm1 # VM1 MODEL
#nnet_egs_dir=exp/xvector_nnet_1a_kadv5/egs
nnet_egs_dir=$nnet_dir/egs

#train_data=train_460_sg400k_s3
#train_plda=train_plda_460_sg400k_s3
#enroll_data=test_clean_enroll_sg400k_s3
#trial_data=test_clean_trial_sg400k_s3
train_data=train_460_vm1
train_plda=train_plda_460_vm1
enroll_data=test_clean_enroll_vm1
trial_data=test_clean_trial_sg_s3

stage=8
if [ $stage -le 0 ]; then

  # format the data as Kaldi data directories
  #for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
  for part in train-clean-100 train-clean-360; do
    # use underscore-separated names in data directories.
    local/data_prep_adv.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done

  # Combine all training data into one
  utils/combine_data.sh data/train_460 \
	  data/train_clean_100 data/train_clean_360

  # Make enrollment and trial data
  #python local/make_librispeech_eval.py ./proto $data/LibriSpeech/test-clean
  #utils/utt2spk_to_spk2utt.pl data/test_clean_enroll/utt2spk > data/test_clean_enroll/spk2utt
  #utils/utt2spk_to_spk2utt.pl data/test_clean_trial/utt2spk > data/test_clean_trial/spk2utt
fi
#exit 0

nj=29
if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  #for name in dev_clean test_clean dev_other test_other train_960 test_clean_enroll test_clean_trial; do
  #for name in train_460 test_clean_enroll test_clean_trial; do
  #for name in train_460_vm1; do
  for name in ${train_data} ${enroll_data} ${trial_data} ; do
  #for name in ${enroll_data} ${trial_data} ; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $nj --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

nj=48

# In this section, we augment the SWBD and SRE data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.  The SRE
# subset will be used to train the PLDA model.

if [ $stage -le 2 ]; then
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

#exit 0

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 3 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj $nj --cmd "$train_cmd" \
	  data/${train_data}_combined data/${train_data}_combined_no_sil exp/${train_data}_combined_no_sil
  utils/fix_data_dir.sh data/${train_data}_combined_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=500
  mv data/${train_data}_combined_no_sil/utt2num_frames data/${train_data}_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/${train_data}_combined_no_sil/utt2num_frames.bak > data/${train_data}_combined_no_sil/utt2num_frames
  utils/filter_scp.pl data/${train_data}_combined_no_sil/utt2num_frames data/${train_data}_combined_no_sil/utt2spk > data/${train_data}_combined_no_sil/utt2spk.new
  mv data/${train_data}_combined_no_sil/utt2spk.new data/${train_data}_combined_no_sil/utt2spk
  utils/fix_data_dir.sh data/${train_data}_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/${train_data}_combined_no_sil/spk2utt > data/${train_data}_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/${train_data}_combined_no_sil/spk2num | utils/filter_scp.pl - data/${train_data}_combined_no_sil/spk2utt > data/${train_data}_combined_no_sil/spk2utt.new
  mv data/${train_data}_combined_no_sil/spk2utt.new data/${train_data}_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/${train_data}_combined_no_sil/spk2utt > data/${train_data}_combined_no_sil/utt2spk

  utils/filter_scp.pl data/${train_data}_combined_no_sil/utt2spk data/${train_data}_combined_no_sil/utt2num_frames > data/${train_data}_combined_no_sil/utt2num_frames.new
  mv data/${train_data}_combined_no_sil/utt2num_frames.new data/${train_data}_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/${train_data}_combined_no_sil
fi

#exit 0

xvector_train_data=${train_data}_combined_no_sil

local/nnet3/xvector/run_xvector_rm457.sh --stage $stage --train-stage -1 \
  --data data/${xvector_train_data} --nnet-dir $nnet_dir \
  --egs-dir $nnet_egs_dir

nj=48

if [ $stage -le 8 ]; then
  # The SRE16 major is an unlabeled dataset consisting of Cantonese and
  # and Tagalog.  This is useful for things like centering, whitening and
  # score normalization.
  : '
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
    $nnet_dir data/${xvector_train_data} \
    exp/xvectors_${xvector_train_data}

  # Extract xvectors for SRE data (includes Mixer 6). We will use this for
  # things like LDA or PLDA.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj $nj \
    $nnet_dir data/${train_plda} \
    exp/xvectors_${train_plda}
  '

  # The SRE16 test data
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 29 \
    $nnet_dir data/${trial_data} \
    exp/xvectors_${trial_data}

  : '
  # The SRE16 enroll data
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 29 \
    $nnet_dir data/${enroll_data} \
    exp/xvectors_${enroll_data}
  '
fi

if [ $stage -le 9 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd exp/xvectors_${xvector_train_data}/log/compute_mean.log \
    ivector-mean scp:exp/xvectors_${xvector_train_data}/xvector.scp \
    exp/xvectors_${xvector_train_data}/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=150
  $train_cmd exp/xvectors_${train_plda}/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/xvectors_${train_plda}/xvector.scp ark:- |" \
    ark:data/${train_plda}/utt2spk exp/xvectors_${train_plda}/transform.mat || exit 1;

  # Train an out-of-domain PLDA model.
  $train_cmd exp/xvectors_${train_plda}/log/plda.log \
    ivector-compute-plda ark:data/${train_plda}/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/xvectors_${train_plda}/xvector.scp ark:- | transform-vec exp/xvectors_${train_plda}/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/xvectors_${train_plda}/plda || exit 1;

  # Here we adapt the out-of-domain PLDA model to SRE16 major, a pile
  # of unlabeled in-domain data.  In the future, we will include a clustering
  # based approach for domain adaptation, which tends to work better.
  $train_cmd exp/xvectors_${xvector_train_data}/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    exp/xvectors_${train_plda}/plda \
    "ark:ivector-subtract-global-mean scp:exp/xvectors_${xvector_train_data}/xvector.scp ark:- | transform-vec exp/xvectors_${train_plda}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    exp/xvectors_${xvector_train_data}/plda_adapt || exit 1;
fi

if [ $stage -le 10 ]; then
  # Get results using the out-of-domain PLDA model.
  $train_cmd exp/scores/log/erep_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/xvectors_${enroll_data}/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/xvectors_${train_plda}/plda - |" \
    "ark:ivector-mean ark:data/${enroll_data}/spk2utt scp:exp/xvectors_${enroll_data}/xvector.scp ark:- | ivector-subtract-global-mean exp/xvectors_${xvector_train_data}/mean.vec ark:- ark:- | transform-vec exp/xvectors_${train_plda}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/xvectors_${xvector_train_data}/mean.vec scp:exp/xvectors_${trial_data}/xvector.scp ark:- | transform-vec exp/xvectors_${train_plda}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" exp/scores/erep_eval_scores || exit 1;

  utils/filter_scp.pl $sre16_trials_tgl exp/scores/erep_eval_scores > exp/scores/erep_eval_tgl_scores
  utils/filter_scp.pl $sre16_trials_yue exp/scores/erep_eval_scores > exp/scores/erep_eval_yue_scores
  pooled_eer=$(paste $sre16_trials exp/scores/erep_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  tgl_eer=$(paste $sre16_trials_tgl exp/scores/erep_eval_tgl_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  yue_eer=$(paste $sre16_trials_yue exp/scores/erep_eval_yue_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Out-of-Domain PLDA, EER: Pooled ${pooled_eer}%, Male ${tgl_eer}%, Female ${yue_eer}%"
  # EER: Pooled 11.73%, Tagalog 15.96%, Cantonese 7.52%
  # For reference, here's the ivector system from ../v1:
  # EER: Pooled 13.65%, Tagalog 17.73%, Cantonese 9.61%
fi

if [ $stage -le 11 ]; then
  # Get results using the adapted PLDA model.
  $train_cmd exp/scores/log/erep_eval_scoring_adapt.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/xvectors_${enroll_data}/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/xvectors_${xvector_train_data}/plda_adapt - |" \
    "ark:ivector-mean ark:data/${enroll_data}/spk2utt scp:exp/xvectors_${enroll_data}/xvector.scp ark:- | ivector-subtract-global-mean exp/xvectors_${xvector_train_data}/mean.vec ark:- ark:- | transform-vec exp/xvectors_${train_plda}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/xvectors_${xvector_train_data}/mean.vec scp:exp/xvectors_${trial_data}/xvector.scp ark:- | transform-vec exp/xvectors_${train_plda}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" exp/scores/erep_eval_scores_adapt || exit 1;

  utils/filter_scp.pl $sre16_trials_tgl exp/scores/erep_eval_scores_adapt > exp/scores/erep_eval_tgl_scores_adapt
  utils/filter_scp.pl $sre16_trials_yue exp/scores/erep_eval_scores_adapt > exp/scores/erep_eval_yue_scores_adapt
  pooled_eer=$(paste $sre16_trials exp/scores/erep_eval_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  tgl_eer=$(paste $sre16_trials_tgl exp/scores/erep_eval_tgl_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  yue_eer=$(paste $sre16_trials_yue exp/scores/erep_eval_yue_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Adapted PLDA, EER: Pooled ${pooled_eer}%, Male ${tgl_eer}%, Female ${yue_eer}%"
  # EER: Pooled 8.57%, Tagalog 12.29%, Cantonese 4.89%
  # For reference, here's the ivector system from ../v1:
  # EER: Pooled 12.98%, Tagalog 17.8%, Cantonese 8.35%
  #
  # Using the official SRE16 scoring software, we obtain the following equalized results:
  #
  # -- Pooled --
  #  EER:          8.66
  #  min_Cprimary: 0.61
  #  act_Cprimary: 0.62
  #
  # -- Cantonese --
  # EER:           4.69
  # min_Cprimary:  0.42
  # act_Cprimary:  0.43
  #
  # -- Tagalog --
  # EER:          12.63
  # min_Cprimary:  0.76
  # act_Cprimary:  0.81
fi
