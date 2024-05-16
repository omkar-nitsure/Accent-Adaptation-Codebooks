#!/usr/bin/bash

export OPENBLAS_NUM_THREADS=1

feat_dir="/ASR_Kmeans/feats_labs/target-pools/feats_iter4"
splits=("train_aus" "train_can" "train_en" "train_sco" "train_us" "train")
nshard=1
km_paths=("/ASR_Kmeans/models/target-pool/kmeans/aus.sav" "/ASR_Kmeans/models/target-pool/kmeans/can.sav" "/ASR_Kmeans/models/target-pool/kmeans/en.sav" "/ASR_Kmeans/models/target-pool/kmeans/sco.sav" "/ASR_Kmeans/models/target-pool/kmeans/us.sav" "/ASR_Kmeans/models/target-pool/kmeans/all.sav")
n_clusters=(50 50 50 50 50 250)

for i in ${!splits[@]}; do
    python learn_kmeans.py ${feat_dir} ${splits[$i]} ${nshard} ${km_paths[$i]} ${n_clusters[$i]} --percent 0.1
done