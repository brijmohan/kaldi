#!/bin/bash

. path.sh
. cmd.sh

data=train_clean_460
original_data_dir=data/${data}

data_dir=data/${data}_hires
ivec_extractor=exp/nnet3_cleaned/extractor
ivec_data_dir=exp/nnet3_cleaned/ivector_${data}

tree_dir=exp/chain_cleaned/tree_sp
model_dir=exp/chain_cleaned/tdnn_1d_sp
lang_dir=data/lang_chain

ppg_dir=exp/nnet3_cleaned/ppg_wpd_fs1_${data}

nj=32

stage=2

if [ $stage -le 1 ]; then
  utils/copy_data_dir.sh ${original_data_dir} ${data_dir}
  steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
	--cmd "$train_cmd" ${data_dir}

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
       	${data_dir} ${ivec_extractor} ${ivec_data_dir} 
fi

if [ $stage -le 2 ]; then
  steps/nnet3/chain/get_phone_post.sh --cmd "$decode_cmd" --nj $nj \
       	--remove-word-position-dependency false --online-ivector-dir ${ivec_data_dir} \
	${tree_dir} ${model_dir} ${lang_dir} ${data_dir} ${ppg_dir}
fi
