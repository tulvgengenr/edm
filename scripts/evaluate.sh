#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

ckpt='/home/v-lijunzhe/blob/sfmexpresults/junzhe/psm/edm_test/transfer_with_ism_v0/00000-cifar10-32x32-cond-ddpmpp-edm_ism-gpus8-batch512-fp32/network-snapshot-040141.pkl'
sample_outdir='edm/fid-tmp/finetune_40141'
ref_npz='edm/fid-refs/cifar10-32x32.npz'

mkdir -p $sample_outdir

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

# Generate 50000 images and save them as fid-tmp/*/*.png
torchrun --standalone --nproc_per_node=1 edm/generate.py --outdir=$sample_outdir --seeds=0-49999 --subdirs --network=$ckpt

# Calculate FID
torchrun --standalone --nproc_per_node=1 edm/fid.py calc --images=$sample_outdir --ref=$ref_npz

sleep infinity
