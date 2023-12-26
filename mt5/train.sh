#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2
for n in {1..3}
do
    BATCH_SIZE=$((2**n))
    EPOCH_SIZE=$((5*n))
    echo "Running experiment with BATCH_SIZE=$BATCH_SIZE"
    python train.py \
        --batch_size $BATCH_SIZE \
        --epoch $EPOCH_SIZE
done

for n in {1..3}
do
    BATCH_SIZE=$((2**n))
    EPOCH_SIZE=$((5*n))
    echo "Running experiment with BATCH_SIZE=$BATCH_SIZE"
    python train.py \
        --batch_size $BATCH_SIZE \
        --epoch $EPOCH_SIZE \
        --
done