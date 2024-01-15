#!/usr/bin/bash
MODEL=EfficientPunct
DEVICES=0
export CUDA_VISIBLE_DEVICES=${DEVICES}

nohup python -u run.py \
--device cuda \
--config_path configs/$MODEL.json \
--data_path data/mustcv1/ \
--load_ckpt models/trained/$MODEL/EfficientPunct.pt \
--save_path models/trained/$MODEL/ \
--mode predict \
--optimizer sgd \
--num_workers 2 \
--batch_size 16 \
--lr 1e-5 \
--epochs 5 \
--save_freq 10 \
> run.log 2>&1 &