#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=1 python fairseq_cli/hydra_train.py \
  --config-dir /ASR_Kmeans/codebase/target-pool/examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=/ASR_Kmeans/feats_labs/target-pools task.label_dir=/ASR_Kmeans/feats_labs/target-pools/labels_iter4 task.labels='["km"]' model.label_rate=50