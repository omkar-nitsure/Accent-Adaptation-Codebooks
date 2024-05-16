#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=3 python fairseq_cli/hydra_train.py \
  --config-dir /ASR_Kmeans/codebase/codebooks_budget/examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=/ASR_Kmeans/feats_labs/budget_loss task.label_dir=/ASR_Kmeans/feats_labs/budget_loss/labels_iter3 task.labels='["km"]' model.label_rate=50