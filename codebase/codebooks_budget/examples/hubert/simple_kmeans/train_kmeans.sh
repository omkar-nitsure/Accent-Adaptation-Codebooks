#!/usr/bin/bash

export OPENBLAS_NUM_THREADS=1

feat_dir="/ASR_Kmeans/data/generic-codebooks/feats_iter4"
splits="train"
nshard=1
km_path="/ASR_Kmeans/models/accent-generic-codebooks/kmeans/iter4/kmeans_model.sav"
n_clusters=500

python learn_kmeans.py ${feat_dir} ${splits} ${nshard} ${km_path} ${n_clusters} --percent 0.1

