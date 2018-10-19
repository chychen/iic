#!/bin/bash
python train.py \
--comment="first try with well coded program" \
--train_dir="log/v2" \
--num_gpus=4 \
--buffer_size=100000 \
--num_threads=40 \
--batch_size=256
