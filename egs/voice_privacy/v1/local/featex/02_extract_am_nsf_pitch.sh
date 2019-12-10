#!/bin/bash

. path.sh
. cmd.sh

nj=20

. utils/parse_options.sh

if [ $# != 2 ]; then
  echo "Usage: "
  echo "  $0 [options] <split-data_dir> <split-data>"
  echo "Options"
  echo "   --nj=40     # Number of CPUs to use for feature extraction"
  exit 1;
fi

data_root=$1
data_dirs=$2

for data in $data_dirs; do
  data_dir=${data_root}/${data}
  pitch_dir=${data_dir}/pitch
  local/featex/make_pitch.sh --nj $nj --cmd "$train_cmd" ${data_dir} \
       exp/make_pitch ${pitch_dir}
done
