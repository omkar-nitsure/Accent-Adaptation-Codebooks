#!/usr/bin/bash

tsv_dir="/ASR_Kmeans/data/accent-targets"
ckpt_path="/ASR_Kmeans/models/accent-targets/train_logs/checkpoint_23_200000.pt"
## splits=("train_aus" "train_can" "train_en" "train_sco" "train_us" "train_without" "val_aus" "val_can" "val_en" "val_sco" "val_us" "valid_without")

splits=("train_without" "valid_without")
nshard=1
layer=6
rank=0
feat_dir="/ASR_Kmeans/feats1"

for i in ${!splits[@]}; do
    python dump_hubert_feature.py ${tsv_dir} ${splits[$i]} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}
done