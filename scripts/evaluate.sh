#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

[ -z "${ckpt}" ] && ckpt='/home/v-lijunzhe/blob/sfmexpresults/junzhe/psm/edm_test/baseline/00004-cifar10-32x32-cond-ddpmpp-edm-gpus8-batch512-fp32/network-snapshot-200000.pkl'
[ -z "${sample_outdir}" ] && sample_outdir='edm/fid-tmp/baseline'
[ -z "${ref_npz}" ] && ref_npz='edm/fid-refs/cifar10-32x32.npz'

mkdir -p $sample_outdir

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

# Generate 50000 images and save them as fid-tmp/*/*.png
torchrun --standalone --nproc_per_node=1 edm/generate.py --outdir=$sample_outdir --seeds=0-49999 --subdirs --network=$ckpt

# Calculate FID
torchrun --standalone --nproc_per_node=1 edm/fid.py calc --images=$sample_outdir --ref=$ref_npz

sleep infinity
