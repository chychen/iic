#!/bin/bash
python train.py \
--comment="resize after norm" \
--train_dir="log/v4" \
--num_gpus=3 \
--buffer_size=10000 \
--num_threads=20 \
--batch_size=128
