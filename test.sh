#!/bin/bash

# eval on hicodet(default) or vg
CUDA_VISIBLE_DEVICES=3 python main.py \
      --eval \
      --port 2236 \
      --resume checkpoints/dhd.pt \
      --output-dir checkpoints/test \
      --model dhd \
      --use_cache_box \
#      --training_type ua \
#      --dataset vg
