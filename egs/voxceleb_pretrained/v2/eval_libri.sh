#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

nnet_dir=exp/0007_voxceleb_v2_1a/exp/xvector_nnet_1a # Pretrained model downloaded from Kaldi website

stage=4

libri_enroll=dev_clean_enroll
libri_trials=dev_clean_trial
librispeech_trials_file=data/$libri_trials/trials
libri_male=${librispeech_trials_file}_male
libri_female=${librispeech_trials_file}_female


if [ $stage -le 0 ]; then
  python local/make_librispeech_eval2.py proto/ASV_VoicePrivacy_v0 /home/bsrivast/asr_data/LibriSpeech || exit 1;
  for name in $libri_enroll $libri_trials; do
    for f in `ls data/${name}`; do
      mv data/${name}/$f data/${name}/${f}.u
      sort -u data/${name}/${f}.u > data/${name}/$f
      rm data/${name}/${f}.u
    done
    utils/utt2spk_to_spk2utt.pl data/${name}/utt2spk > data/${name}/spk2utt
  done
fi

# Evaluating LibriSpeech trials using VoXceleb pretrained model 
# without adaptation

nj=29
if [ $stage -le 1 ]; then
  echo "Evaluating LibriSpeech trials using pretrained VoXceleb model."

  echo "Compute MFCC..."
  for name in $libri_enroll $libri_trials; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $nj --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    
    utils/fix_data_dir.sh data/${name}
    
    sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done

fi

if [ $stage -le 2 ]; then
  echo "Extract xvectors..."
  for name in $libri_enroll $libri_trials; do
    sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj $nj \
      $nnet_dir data/${name} \
      $nnet_dir/xvectors_${name}
  done
fi

if [ $stage -le 3 ]; then
  echo "Scoring the trials..."
  $train_cmd exp/scores/log/librispeech_trial_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_${libri_enroll}/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train/plda - |" \
    "ark:ivector-mean ark:data/${libri_enroll}/spk2utt scp:${nnet_dir}/xvectors_${libri_enroll}/xvector.scp ark:- | ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec ark:- ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$nnet_dir/xvectors_${libri_trials}/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$librispeech_trials_file' | cut -d\  --fields=1,2 |" exp/scores_libri_trials || exit 1;

  utils/filter_scp.pl $libri_male exp/scores_libri_trials > exp/scores_libri_male
  utils/filter_scp.pl $libri_female exp/scores_libri_trials > exp/scores_libri_female
  pooled_eer=$(paste $librispeech_trials_file exp/scores_libri_trials | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  male_eer=$(paste $libri_male exp/scores_libri_male | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  female_eer=$(paste $libri_female exp/scores_libri_female | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "EER: Pooled ${pooled_eer}%, Male ${male_eer}%, Female ${female_eer}%"
fi

# Evaluating LibriSpeech trials using VoXceleb pretrained model 
# WITH adaptation

adapt_data=dev_other
if [ $stage -le 4 ]; then
  # Prepare adaptation data
  echo "stage 16: extracting MFCC for adaptation data..."
  steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $nj --cmd "$train_cmd" \
      data/${adapt_data} exp/make_mfcc $mfccdir
    
  utils/fix_data_dir.sh data/${adapt_data}
    
  sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
      data/${adapt_data} exp/make_vad $vaddir
  utils/fix_data_dir.sh data/${adapt_data}
fi

if [ $stage -le 5 ]; then
  echo "Extract xvectors for adaptation data..."
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj $nj \
      $nnet_dir data/${adapt_data} \
      $nnet_dir/xvectors_${adapt_data}
fi

# DEV_OTHER ADAPTATION
if [ $stage -le 6 ]; then
  # Here we adapt the out-of-domain VoxCeleb PLDA model to LibriSpeech dev_other.  In the future, we will include a clustering
  # based approach for domain adaptation, which tends to work better.
  echo "Adapting the VoxCeleb model to dev_other..."
  $train_cmd $nnet_dir/xvectors_${adapt_data}/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    $nnet_dir/xvectors_train/plda \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/xvectors_${adapt_data}/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${nnet_dir}/xvectors_${adapt_data}/plda_adapt || exit 1;
fi

if [ $stage -le 7 ]; then
  echo "Scoring the trials with dev_other adapted VoxCeleb model..."
  $train_cmd exp/scores/log/librispeech_${adapt_data}_adapt.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_${libri_enroll}/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_${adapt_data}/plda_adapt - |" \
    "ark:ivector-mean ark:data/${libri_enroll}/spk2utt scp:${nnet_dir}/xvectors_${libri_enroll}/xvector.scp ark:- | ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec ark:- ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$nnet_dir/xvectors_${libri_trials}/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$librispeech_trials_file' | cut -d\  --fields=1,2 |" exp/scores/libri_${adapt_data}_adapt || exit 1;

  utils/filter_scp.pl $libri_male exp/scores/libri_${adapt_data}_adapt > exp/scores/libri_${adapt_data}_male
  utils/filter_scp.pl $libri_female exp/scores/libri_${adapt_data}_adapt > exp/scores/libri_${adapt_data}_female
  pooled_eer=$(paste $librispeech_trials_file exp/scores/libri_${adapt_data}_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  male_eer=$(paste $libri_male exp/scores/libri_${adapt_data}_male | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  female_eer=$(paste $libri_female exp/scores/libri_${adapt_data}_female | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "EER: Pooled ${pooled_eer}%, Male ${male_eer}%, Female ${female_eer}%"
fi

