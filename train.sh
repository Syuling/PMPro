#!/bin/bash


#  =================
DATASET=$1  # oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet
BETA=$2 #$ # loss regularization
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)

#  =================
CFG=rn50

for SEED in 1 2 3
do
   	python train.py \
        --dataset ${DATASET} \
    	--seed ${SEED} \
    	--train-shot ${SHOTS} \
        --beta ${BETA}
done