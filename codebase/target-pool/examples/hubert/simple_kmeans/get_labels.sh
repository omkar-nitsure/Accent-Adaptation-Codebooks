#!/usr/bin/bash

feat_dir="/ASR_Kmeans/feats_labs/target-pools/feats_iter4"
splits=("train" "valid")
km_path="/ASR_Kmeans/models/target-pool/kmeans/iter4.sav"
rank=0
nshard=1
lab_dir="/ASR_Kmeans/feats_labs/target-pools/labels_iter4"

for i in ${!splits[@]}; do
    python dump_km_label.py ${feat_dir} ${splits[$i]} ${km_path} ${nshard} ${rank} ${lab_dir}
done

n_clusters=500
for x in $(seq 0 $((n_clusters - 1))); do
  echo "$x 1"
done >> $lab_dir/dict.km.txt

# n_clusters=270
# for x in $(seq 0 $((n_clusters - 1))); do
#   echo "$x 1"
# done >> $lab_dir/dict_270.km.txt
