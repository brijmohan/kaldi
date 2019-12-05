#!/bin/bash

. path.sh
. cmd.sh

data_root=data/am_nsf
data_dirs="train_clean_460_train train_clean_460_dev train_clean_460_test"
nj=40

for data in $data_dirs; do
  data_dir=${data_root}/${data}
  pitch_dir=${data_dir}/pitch
  local/make_pitch.sh --nj $nj --cmd "$train_cmd" ${data_dir} \
       exp/make_pitch ${pitch_dir}
done
