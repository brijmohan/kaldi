#!/bin/bash

train_data=data/am_nsf/train_clean_460_train
dev_data=data/am_nsf/train_clean_460_dev
test_data=data/am_nsf/train_clean_460_test

ppg_file=exp/nnet3_cleaned/ppg_wpd_fs1_train_clean_460/phone_post.scp
melspec_file=data/train_clean_460_mspec/feats.scp

out_dir=/media/data/am_nsf_data/librispeech/train_460

: '
echo "Writing SCP files.."
cut -f 1 -d' ' ${train_data}/utt2spk > ${out_dir}/scp/train.lst
cut -f 1 -d' ' ${dev_data}/utt2spk > ${out_dir}/scp/dev.lst
cut -f 1 -d' ' ${test_data}/utt2spk > ${out_dir}/scp/test.lst

python local/am_nsf/create_ppg_melspec_data.py ${ppg_file} ${melspec_file} ${out_dir}
'

echo "Writing xvector and F0 for train."
python local/am_nsf/create_xvector_f0_data.py ${train_data} ${out_dir}
echo "Writing xvector and F0 for dev."
python local/am_nsf/create_xvector_f0_data.py ${dev_data} ${out_dir}
echo "Writing xvector and F0 for test."
python local/am_nsf/create_xvector_f0_data.py ${test_data} ${out_dir}
