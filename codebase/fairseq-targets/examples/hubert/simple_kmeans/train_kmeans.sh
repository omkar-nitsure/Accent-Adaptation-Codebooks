#!/usr/bin/bash

export OPENBLAS_NUM_THREADS=1

feat_dir="/ASR_Kmeans/feats1"
splits=("train_aus" "train_can" "train_en" "train_sco" "train_us" "train_without")
nshard=1
km_paths=("/ASR_Kmeans/models/accent-targets/kmeans/iter4/aus.sav" "/ASR_Kmeans/models/accent-targets/kmeans/iter4/can.sav" "/ASR_Kmeans/models/accent-targets/kmeans/iter4/en.sav" "/ASR_Kmeans/models/accent-targets/kmeans/iter4/sco.sav" "/ASR_Kmeans/models/accent-targets/kmeans/iter4/us.sav" "/ASR_Kmeans/models/accent-targets/kmeans/iter4/all.sav")
n_clusters=(50 50 50 50 50 250)

for i in ${!splits[@]}; do
    python learn_kmeans.py ${feat_dir} ${splits[$i]} ${nshard} ${km_paths[$i]} ${n_clusters[$i]} --percent 0.1
done