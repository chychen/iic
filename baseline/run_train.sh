#!/bin/bash
python train.py \
--train_dir="log/v1" \
--num_gpus=1 \
--buffer_size=10000 \
--num_threads=20 \
--batch_size=64