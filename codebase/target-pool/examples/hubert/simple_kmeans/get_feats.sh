#!/usr/bin/bash

tsv_dir="/ASR_Kmeans/data/clean_tsv"
ckpt_path="/ASR_Kmeans/models/target-pool/iter3/checkpoint_best.pt"
splits=("train_aus" "train_can" "train_en" "train_sco" "train_us" "train" "val_aus" "val_can" "val_en" "val_sco" "val_us" "valid")
nshard=1
layer=6
rank=0
feat_dir="/ASR_Kmeans/feats_labs/target-pools/feats_iter4"

for i in ${!splits[@]}; do
    python dump_hubert_feature.py ${tsv_dir} ${splits[$i]} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}
done
