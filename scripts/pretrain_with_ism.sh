#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

outdir='/sfm/sfmexpresults/junzhe/psm/edm_test/with_ism_v2'
n_gpu=$(nvidia-smi -L | wc -l)
echo "n_gpu: ${n_gpu}"

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

torchrun --standalone --nproc_per_node=$n_gpu edm/train.py --precond=edm_ism --outdir=$outdir --data=edm/datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp --ism_weight=0.1 --ism_rng_mean=-3.0 --ism_dy=1e-5 --batch=512

sleep infinity
