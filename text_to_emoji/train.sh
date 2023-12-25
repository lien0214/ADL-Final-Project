#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2
for n in {1..8}
do
    BATCH_SIZE=$((4*n))
    echo "Running experiment with BATCH_SIZE=$BATCH_SIZE"
    python train.py --batch_size $BATCH_SIZE
done