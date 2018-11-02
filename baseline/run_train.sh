#!/bin/bash
python train.py \
--comment="try autoaugment" \
--train_dir="log/v3" \
--num_gpus=4 \
--buffer_size=10000 \
--num_threads=40 \
--batch_size=256
