#!/usr/bin/bash

feat_dir="/ASR_Kmeans/feats1"
splits=("train_aus" "train_can" "train_en" "train_sco" "train_us" "train_without" "val_aus" "val_can" "val_en" "val_sco" "val_us" "valid_without")
km_paths=("/ASR_Kmeans/models/accent-targets/kmeans/iter4/aus.sav" "/ASR_Kmeans/models/accent-targets/kmeans/iter4/can.sav" "/ASR_Kmeans/models/accent-targets/kmeans/iter4/en.sav" "/ASR_Kmeans/models/accent-targets/kmeans/iter4/sco.sav" "/ASR_Kmeans/models/accent-targets/kmeans/iter4/us.sav" "/ASR_Kmeans/models/accent-targets/kmeans/iter4/all.sav" "/ASR_Kmeans/models/accent-targets/kmeans/iter4/aus.sav" "/ASR_Kmeans/models/accent-targets/kmeans/iter4/can.sav" "/ASR_Kmeans/models/accent-targets/kmeans/iter4/en.sav" "/ASR_Kmeans/models/accent-targets/kmeans/iter4/sco.sav" "/ASR_Kmeans/models/accent-targets/kmeans/iter4/us.sav" "/ASR_Kmeans/models/accent-targets/kmeans/iter4/all.sav")
nshard=1
rank=0
lab_dir="/ASR_Kmeans/labels1"

for i in ${!splits[@]}; do
    python dump_km_label.py ${feat_dir} ${splits[$i]} ${km_paths[$i]} ${nshard} ${rank} ${lab_dir}
done

n_clusters=250
for x in $(seq 0 $((n_clusters - 1))); do
  echo "$x 1"
done >> $lab_dir/dict_250.km.txt

n_clusters=270
for x in $(seq 0 $((n_clusters - 1))); do
  echo "$x 1"
done >> $lab_dir/dict_270.km.txt