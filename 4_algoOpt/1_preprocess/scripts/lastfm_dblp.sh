#!/bin/bash

if [[ $# -ne 1 ]]
then
    echo Usage: ./lastfm_dblp.sh [dataset_name]
    echo [lastfm, dblp]
    echo ex. ./lastfm_dblp.sh lastfm
    exit
fi

mkdir -p $HOME/MERCI/data/2_transactions/$1/
mkdir -p $HOME/MERCI/data/3_train_test/$1/
mkdir -p $HOME/MERCI/data/4_filtered/$1/

# Parse raw dataset
python3 ../src/parse_$1.py $HOME/MERCI/data/1_raw_data/$1/

# Divide train/test dataset
python3 ../src/train_test_division.py $HOME/MERCI/data/2_transactions/$1/$1_transactions.txt
mv ./$1* $HOME/MERCI/data/3_train_test/$1/

# Filter features by occurrence
python3 ../src/filter_by_occurrence.py $1
