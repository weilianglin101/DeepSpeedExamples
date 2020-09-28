#!/bin/bash

# JOB_NAME=adam_bsz1k_lr1e-3_WarmupLR
JOB_NAME=onebitlamb_bsz1k_lr1e-2

# deepspeed cifar10_deepspeed_dist.py --deepspeed --deepspeed_config ds_config_dist.json --epochs 250 --batch_size 128 --job_name $JOB_NAME > output/$JOB_NAME.log
# deepspeed cifar10_deepspeed_dist.py --deepspeed --deepspeed_config ds_config_dist_lamb.json --epochs 250 --batch_size 128 --job_name $JOB_NAME > output/$JOB_NAME.log
deepspeed cifar10_deepspeed_dist.py --deepspeed --deepspeed_config ds_config_dist_onebitlamb.json --epochs 250 --batch_size 128 --job_name $JOB_NAME > output/$JOB_NAME.log
