#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

ckpt='/home/v-lijunzhe/blob/sfmexpresults/junzhe/psm/edm_test/transfer_with_ism_v0/00003-cifar10-32x32-cond-ddpmpp-edm_ism-gpus8-batch512-fp32/network-snapshot-100352.pkl'
sample_outdir='edm/fid-tmp/transfer_with_ism_weight_03_mean_1_100000'
ref_npz='edm/fid-refs/cifar10-32x32.npz'

mkdir -p $sample_outdir

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

# # Generate 50000 images and save them as fid-tmp/*/*.png
# torchrun --standalone --nproc_per_node=1 edm/generate.py --outdir=$sample_outdir --seeds=0-49999 --subdirs --network=$ckpt

# # Calculate FID
# torchrun --standalone --nproc_per_node=1 edm/fid.py calc --images=$sample_outdir --ref=$ref_npz

# sample example 
python edm/example.py --model=$ckpt --num_steps=18 --output=$sample_outdir/example.png

sleep infinity
