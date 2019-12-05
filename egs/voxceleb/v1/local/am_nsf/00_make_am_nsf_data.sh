#!/bin/bash

in_dir=data/train_clean_460
out_dir=data/am_nsf
mkdir -p ${out_dir}

python local/am_nsf/split_am_nsf_data.py ${in_dir} ${out_dir} 40 40

# sort each file
train_dir=$out_dir/$(basename $in_dir)_train
dev_dir=$out_dir/$(basename $in_dir)_dev
test_dir=$out_dir/$(basename $in_dir)_test

echo "Sorting : ${train_dir}, ${dev_dir} and ${test_dir}" 

for f in `ls ${train_dir}`; do
  echo "Sorting $f"
  sort -u ${train_dir}/$f > ${train_dir}/${f%.*}
  rm ${train_dir}/$f
done

for f in `ls ${dev_dir}`; do
  echo "Sorting $f"
  sort -u ${dev_dir}/$f > ${dev_dir}/${f%.*}
  rm ${dev_dir}/$f
done

for f in `ls ${test_dir}`; do
  echo "Sorting $f"
  sort -u ${test_dir}/$f > ${test_dir}/${f%.*}
  rm ${test_dir}/$f
done
