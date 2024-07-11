#!/bin/bash

NPROC_PER_NODE=8
NNODES=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
ROOT_DIR=/mnt/data/group/majian/LLaMA-Factory

torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    $ROOT_DIR/src/train.py $ROOT_DIR/examples/full_multi_gpu/baichuan.yaml
