#!/bin/bash

if [[ $# -ne 2 ]]
then
    echo Usage: ./run_patoh.sh [dataset_name] [num_partitions]
    echo ex. ./run_patoh.sh lastfm 450
    exit
fi

../bin/patoh $HOME/MERCI/data/4_filtered/$1/$1_train_filtered.txt $2 BA=0
mkdir -p $HOME/MERCI/data/5_patoh/$1/partition_$2/
mv $HOME/MERCI/data/4_filtered/$1/$1_train_filtered.txt.part.$2 $HOME/MERCI/data/5_patoh/$1/partition_$2/$1_train_filtered.txt.part.$2 