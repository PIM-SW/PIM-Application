#!/bin/bash

# Dataset
DATASET=("amazon_Home_and_Kitchen" "amazon_Sports_and_Outdoors" "amazon_Electronics" "amazon_Office_Products")
num_partitions=(32000 25360 14880 10992) # partition size = 32
# num_partitions=(8000 6340 3720 2748) # partition size = 128

# Variables
len=${#DATASET[@]}

# Correlation-Aware Variable-Sized Clustering
cd ./3_clustering
mkdir -p bin
make

for (( i=0; i<$len; i++ )); do
    printf "\nCreating remapped mapping on dataset %s\n" ${DATASET[$i]}
    ./bin/clustering -d ${DATASET[$i]} -p ${num_partitions[$i]} --remap-only
done
