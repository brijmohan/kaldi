#!/bin/sh

. path.sh
. local/vc/am/init.sh

proj_dir=${CURRENNT_SCRIPTS}/acoustic-modeling/project-DAR-continuous


# preparing the training data
python ${proj_dir}/../SCRIPTS/01_prepare.py config_libri_am

# training the RNN model
python ${proj_dir}/../SCRIPTS/02_train.py config_libri_am

