#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0 python fairseq_cli/hydra_train.py \
  --config-dir /ASR_Kmeans/codebase/fairseq-codebooks-general/examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=/ASR_Kmeans/feats_labs/generic-codebooks task.label_dir=/ASR_Kmeans/feats_labs/generic-codebooks/labels_iter3 task.labels='["km"]' model.label_rate=50