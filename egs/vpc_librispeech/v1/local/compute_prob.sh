#!/bin/bash

. path.sh

nnet3-compute-prob exp/xvector_nnet_1a_kadv5_rm457/final.raw 'ark,bg:nnet3-copy-egs scp:exp/xvector_nnet_1a_kadv5_rm457/egs/valid_diagnostic_adv.scp ark:- | nnet3-merge-egs --minibatch-size=1:64 ark:- ark:- |'

