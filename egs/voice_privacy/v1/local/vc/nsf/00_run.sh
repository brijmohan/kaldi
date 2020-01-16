#!/bin/sh

# preparing data
python ../SCRIPTS/00_prepare_data.py config_libri

# model training
python ../SCRIPTS/01_train_network.py config_libri
