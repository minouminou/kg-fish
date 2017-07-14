# kg-fish
Code for Kaggle competition [The Nature Conservancy Fisheries Monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring)

Goal of this competition is to detect which species of fish appears on a fishing boat, based on images captured from boat cameras of various angles.
Eight target categories are available in this dataset: Albacore tuna, Bigeye tuna, Yellowfin tuna, Mahi Mahi, Opah, Sharks, Other, and No Fish.
Each image has only one fish category, except that there are sometimes very small fish in the pictures that are used as bait.

![fish](https://raw.githubusercontent.com/minouminou/kg-fish/master/data/data_desc/species-ref-key.jpg)

## Description
DeepMask/SharpMask modified from https://github.com/facebookresearch/deepmask
MultiPathNet modified from https://github.com/facebookresearch/multipathnet

## Requirements
- Linux (Ubuntu 16.04 LTS x86_64)
- NVidia GPU with compute capability 3.5+ and memory 4Gb+
- Python 2.7
- Python libraries
- CUDA and cuDNN v2 (or newer)
- Torch with packages: COCO API, image, tds, cjson, nnx, optim, inn, cutorch, cunn, cudnn

## Training models
### Data
Warning: data contains graphic contents that some may find disturbing.

Unpack data to data/raw directory.
For annotations see data/dataset/annotations/ and image_labeler.
If you need to label new data it must be in MS COCO format.
Directory image_labeler contains code for labeling data and making dataset in Microsoft COCO format.
Labeling code provided as is.
*.segm.txt file row format: x1,x2,y1,y2,...,xn,yn
*.lbl.txt file row format: class

### Train DeepMask
```bash
cd ~/kg-fish/multipathnet/deepmask/
th train.lua -nthreads 1 -batch 8 -seed 42 -gpu 1
cp exps/deepmask/exp,batch=8,nthreads=1,seed=42/bestmodel.t7 ../../data/models/deepmask/model.t7
```

### Train SharpMask
```bash
th train.lua -nthreads 1 -batch 4 -seed 42 -dm ../../data/models/deepmask/model.t7 -gpu 1
```

### Compute mask proposals
Mask proposals based on DeepMask
```bash
th createProposals.lua -split train -startAt 1 -endAt 1240 -seed 42 -gpu 1 -np 1000 -dm -savedir ../../data/proposals/deepmask ../../data/models/deepmask
th createProposals.lua -split val -startAt 1 -endAt 653 -seed 42 -gpu 1 -np 1000 -dm -savedir ../../data/proposals/deepmask ../../data/models/deepmask
```
or Mask proposals based on SharpMask
```bash
th createProposals.lua -split train -startAt 1 -endAt 1240 -seed 42 -gpu 1 -np 1000 -savedir ../../data/proposals/deepmask ../../data/models/deepmask
th createProposals.lua -split val -startAt 1 -endAt 653 -seed 42 -gpu 1 -np 1000 -savedir ../../data/proposals/deepmask ../../data/models/deepmask
```

### Train Multipathnet
```bash
cd ~/kg-fish/multipathnet/
cp -r ../data/dataset/* ./data
cp ../data/proposals/deepmask/* ./data/proposals/coco/deepmask
```
Copy model (see [https://github.com/facebook/fb.resnet.torch/tree/master/pretrained](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained) )
```bash
wget https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7 -O ./data/models/resnet/resnet-18.t7
```
To train MultiPathNet 
```bash
./scripts/fish_train.sh
```

## Test
```bash
th findFish.lua -seed 42 -dm -gpu 1 \
                -sharpmask_path ../data/models/deepmask/model.t7 \
                -multipath_path ../data/models/multipathnet/model_final.t7 \
                -imgdir ../data/raw/test_stg1 \
                -resfile ../data/submission/stg1.txt \
                -saveimg -savedir ../../result
```