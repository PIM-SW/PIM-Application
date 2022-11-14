#!/bin/bash

# Dataset
DATASET=("amazon_Home_and_Kitchen" "amazon_Sports_and_Outdoors" "amazon_Electronics" "amazon_Office_Products")
num_partitions=(8000 6340 3720 2748)

# Variables
len=${#DATASET[@]}

# Performance Evaluation
cd ./4_performance_evaluation
mkdir -p bin
make all

for (( i=0; i<$len; i++ )); do
    printf "\nRunning baseline on dataset %s\n" ${DATASET[$i]}
    ./bin/eval_baseline -d ${DATASET[$i]}    
    printf "\nRunning remapped on dataset %s\n" ${DATASET[$i]}
    ./bin/eval_remapped -d ${DATASET[$i]} -p ${num_partitions[$i]}
done
