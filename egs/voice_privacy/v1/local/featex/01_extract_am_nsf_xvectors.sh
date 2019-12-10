#!/bin/bash
. path.sh
. cmd.sh

nj=40

. utils/parse_options.sh

if [ $# != 4 ]; then
  echo "Usage: "
  echo "  $0 [options] <data-root> <data-dirs> <nnet-dir> <xvector-out-dir>"
  echo "Options"
  echo "   --nj=40     # Number of CPUs to use for feature extraction"
  exit 1;
fi

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
data_root=$1
data_dirs=$2

nnet_dir=$3
out_dir=$4

mkdir -p ${out_dir}

for data in $data_dirs; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $nj --cmd "$train_cmd" \
      $data_root/${data} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh $data_root/${data}
    
    sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
      $data_root/${data} exp/make_vad $vaddir
    utils/fix_data_dir.sh $data_root/${data}

    sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj $nj \
      $nnet_dir $data_root/$data \
      $out_dir/xvectors_$data
done

