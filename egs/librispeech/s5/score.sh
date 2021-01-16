#!/bin/bash

model=chain_cleaned/tdnn_1d_sp
for test in dev_clean test_clean dev_other test_other; do
  for lm in fglarge tglarge tgmed tgsmall; do
    grep WER exp/${model}/decode_${test}_${lm}/wer* | utils/best_wer.sh;
  done
  echo
done
