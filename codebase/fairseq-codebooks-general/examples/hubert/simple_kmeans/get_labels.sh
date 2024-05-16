#!/usr/bin/bash

feat_dir="/ASR_Kmeans/data/generic-codebooks/feats_iter4"
splits=("train" "valid")
km_paths=("/ASR_Kmeans/models/accent-generic-codebooks/kmeans/iter4/kmeans_model.sav" "/ASR_Kmeans/models/accent-generic-codebooks/kmeans/iter4/kmeans_model.sav")
nshard=1
rank=0
lab_dir="/ASR_Kmeans/generic-codebooks/labels_iter4"

for i in ${!splits[@]}; do
    python dump_km_label.py ${feat_dir} ${splits[$i]} ${km_paths[$i]} ${nshard} ${rank} ${lab_dir}
done

n_clusters=500
for x in $(seq 0 $((n_clusters - 1))); do
  echo "$x 1"
done >> $lab_dir/dict.km.txt


