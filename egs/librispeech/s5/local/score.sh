#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#           2014  Guoguo Chen
# Apache 2.0

model=chain_cleaned/tdnn_1d_sp
#model=nnet3_cleaned/tdnn_sp
for test in dev_clean test_clean dev_other test_other; do
  for lm in fglarge tglarge tgmed tgsmall; do
    grep WER exp/${model}/decode_${test}_${lm}/wer* | utils/best_wer.sh;
  done
  echo
done
