#!/bin/bash
. path.sh
. cmd.sh

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
data_root=data/am_nsf
data_dirs="train_clean_460_train train_clean_460_dev train_clean_460_test"

nnet_dir=exp/0007_voxceleb_v2_1a/exp/xvector_nnet_1a # Pretrained model downloaded from Kaldi website
nj=40

for data in $data_dirs; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $nj --cmd "$train_cmd" \
      $data_root/${data} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh $data_root/${data}
    
    sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
      $data_root/${data} exp/make_vad $vaddir
    utils/fix_data_dir.sh $data_root/${data}

    sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj $nj \
      $nnet_dir $data_root/$data \
      $nnet_dir/am_nsf/xvectors_$data
done

