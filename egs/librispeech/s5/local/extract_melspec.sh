#!/bin/bash

. path.sh
. cmd.sh

odata_dir=data/libritts_train_clean_100
data_dir=${odata_dir}_mspec
nj=40
mspec_dir=${data_dir}/mspec

mspec_config=conf/mspec.conf

utils/copy_data_dir.sh ${odata_dir} ${data_dir}

steps/make_fbank.sh --cmd "$train_cmd" --nj $nj \
       	--fbank-config ${mspec_config} ${data_dir} \
       	exp/make_fbank/${data_dir} $mspec_dir
