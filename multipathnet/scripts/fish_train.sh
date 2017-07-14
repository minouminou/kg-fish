#!/bin/bash

export model=resnet
#export model=alexnet
#export model=multipathnet #OOM
#export model=vgg #OOM
#export model=inceptionv3 #no model
#export model=nin #no model

export proposals=deepmask
export year=2014
export train_set=trainval
export test_set=val
export dataset=coco

export nDonkeys=4
export integral=true
export images_per_batch=4
export batchSize=64
export scale=800
export weightDecay=0
export test_best_proposals_number=400
export test_nsamples=-1 #1000
export train_nsamples=-1

export proposals=deepmask
export epochSize=100
export nEpochs=3200
export step=2800

export batchSize=128
export train_nGPU=1
export test_nGPU=1

export save_folder="logs/coco_${model}_${proposals}_$RANDOM$RANDOM"

mkdir -p $save_folder

th train.lua | tee $save_folder/log.txt

