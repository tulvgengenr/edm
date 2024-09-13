#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

outdir='/sfm/sfmexpresults/junzhe/psm/edm_test/baseline'
n_gpu=$(nvidia-smi -L | wc -l)
echo "n_gpu: ${n_gpu}"

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

torchrun --standalone --nproc_per_node=$n_gpu edm/train.py --outdir=$outdir --data=edm/datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp

sleep infinity
