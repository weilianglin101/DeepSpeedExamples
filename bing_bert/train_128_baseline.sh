#!/bin/bash

base_dir=`pwd`
local_dir=/cl
nfs_dir=/home/conglli
th_dir=/turing-hdd/users/conglli

# Where should we save checkpoints and tensorboard events?
JOB_NAME=$1
#OUTPUT_DIR=${base_dir}/bert_model_outputs
#OUTPUT_DIR=${local_dir}/bert_model_outputs
#OUTPUT_DIR=${nfs_dir}/bert_model_outputs
OUTPUT_DIR=${th_dir}/bert_model_outputs

mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/bert_large_lamb.json \
--max_seq_length 128 \
--output_dir $OUTPUT_DIR \
--deepspeed \
--deepspeed_transformer_kernel \
--print_steps 100 \
--lr_schedule "EE" \
--lr_offset 10e-4 \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz64k_lamb_config_seq128.json \
--data_path_prefix /data/bert \
&> $OUTPUT_DIR/${JOB_NAME}.log
#--ckpt_to_save 150 \
