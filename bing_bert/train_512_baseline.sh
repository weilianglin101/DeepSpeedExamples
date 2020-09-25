#!/bin/bash

base_dir=`pwd`
local_dir=/cl
nfs_dir=/home/conglli
th_dir=/turing-hdd/users/conglli

# Where should we save checkpoints and tensorboard events?
JOB_NAME=$1
CONFIG_NAME=$2
CHECKPOINT_FILE=$3
EPOCH=$4
#OUTPUT_DIR=${base_dir}/bert_model_outputs
#OUTPUT_DIR=${local_dir}/bert_model_outputs
#OUTPUT_DIR=${nfs_dir}/bert_model_outputs
OUTPUT_DIR=${th_dir}/bert_model_outputs
CKPT_DIR=${th_dir}/bert_model_outputs

# Assumes job name in previous seq128 run, will resume training from epoch 150
CHECKPOINT_BASE_PATH=${CKPT_DIR}/saved_models/${CHECKPOINT_FILE}
CHECKPOINT_NAME=`basename ${CHECKPOINT_BASE_PATH}/epoch${EPOCH}_*`
echo "checkpoint id: $CHECKPOINT_NAME"

mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/bert_large_lamb.json \
--max_seq_length 512 \
--output_dir $OUTPUT_DIR \
--print_steps 100 \
--deepspeed \
--deepspeed_transformer_kernel \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/${CONFIG_NAME}.json \
--data_path_prefix /data/bert \
--validation_data_path_prefix /data/bert \
--rewarmup \
--lr_schedule "EE" \
--attention_dropout_checkpoint \
--lr_offset 0.0 \
--load_training_checkpoint ${CHECKPOINT_BASE_PATH} \
--load_checkpoint_id ${CHECKPOINT_NAME} \
&> $OUTPUT_DIR/${JOB_NAME}.log
