#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=7 python fairseq_cli/hydra_train.py \
  --config-dir /ASR_Kmeans/fairseq-targets/examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=/ASR_Kmeans/data/accent-targets task.label_dir=/ASR_Kmeans/accent-targets/labels1 task.labels='["km"]' model.label_rate=50