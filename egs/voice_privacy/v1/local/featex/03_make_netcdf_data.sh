#!/bin/bash

. path.sh
. cmd.sh

stage=0

. utils/parse_options.sh

if [ $# != 8 ]; then
  echo "Usage: "
  echo "  $0 [options] <train-dir> <dev-dir> <test-dir> <ppg-file> <melspec-file> <xvec-out-dir> <out-dir> <test-dir>"
  echo "Options"
  echo "   --stage 0     # Number of CPUs to use for feature extraction"
  exit 1;
fi

train_data=$1
dev_data=$2
test_data=$3

ppg_file=$4
melspec_file=$5
xvec_out_dir=$6

out_dir=$7
test_dir=$8


if [ $stage -le 0 ]; then
  mkdir -p $out_dir/scp $out_dir/xvector $out_dir/f0 $out_dir/ppg $out_dir/mel

  echo "Writing SCP files.."
  cut -f 1 -d' ' ${train_data}/utt2spk > ${out_dir}/scp/train.lst || exit 1;
  cut -f 1 -d' ' ${dev_data}/utt2spk > ${out_dir}/scp/dev.lst || exit 1;
  cut -f 1 -d' ' ${test_data}/utt2spk > ${out_dir}/scp/test.lst || exit 1;
fi


if [ $stage -le 1 ]; then
  python local/featex/create_ppg_melspec_data.py ${ppg_file} ${melspec_file} ${out_dir} || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "Writing xvector and F0 for train."
  python local/featex/create_xvector_f0_data.py ${train_data} ${xvec_out_dir} ${out_dir} || exit 1;
  echo "Writing xvector and F0 for dev."
  python local/featex/create_xvector_f0_data.py ${dev_data} ${xvec_out_dir} ${out_dir} || exit 1;
  echo "Writing xvector and F0 for test."
  python local/featex/create_xvector_f0_data.py ${test_data} ${xvec_out_dir} ${out_dir} || exit 1;
fi

if [ $stage -le 3 ]; then
  echo "Splitting test data in separate folder..."
  python local/featex/split_test_data.py ${out_dir} ${test_dir} || exit 1;
fi
