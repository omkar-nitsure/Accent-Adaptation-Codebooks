#!/usr/bin/bash

tsv_dir="/ASR_Kmeans/data/generic-codebooks"
ckpt_path="/ASR_Kmeans/models/accent-generic-codebooks/iter3/checkpoint_best.pt"

splits=("train" "valid")
nshard=1
layer=6
rank=0
feat_dir="/ASR_Kmeans/data/generic-codebooks/feats_iter4"

for i in ${!splits[@]}; do
    python dump_hubert_feature.py ${tsv_dir} ${splits[$i]} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}
done
