# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import numpy as np

import tqdm
from npy_append_array import NpyAppendArray

accent_to_idx = {"australia": 0, "canada": 1, "england": 2, "scotland": 3, "us": 4}

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("feature_utils")


def get_shard_range(tot, nshard, rank):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(
        f"rank {rank} of {nshard}, process {end-start} " f"({start}-{end}) out of {tot}"
    )
    return start, end


def get_path_iterator(tsv, nshard, rank):
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start:end]

        def iterate():
            for line in lines:
                subpath, nsample, accent = line.split("\t")
                yield f"{root}/{subpath}", int(nsample), accent

    return iterate, len(lines)


# OMKAR
def dump_feature(reader, generator, num, split, nshard, rank, feat_dir):
    iterator = generator()

    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    outs_path = f"/ASR_Kmeans/data/kmeans_audio/outputs/{split}_{rank}_{nshard}.pkl"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"

    os.makedirs(feat_dir, exist_ok=True)
    if os.path.exists(feat_path):
        os.remove(feat_path)

    feat_f = NpyAppendArray(feat_path)
    outs_f = []
    with open(leng_path, "w") as leng_f:
        for path, nsample, accent in tqdm.tqdm(iterator, total=num):
            feat, outs_ = reader.get_feats(path, nsample, [accent_to_idx[accent]])
            feat_f.append(feat.cpu().numpy())
            outs_f.append(outs_.cpu().numpy())
            leng_f.write(f"{len(feat)}\n")

    # Writing the list of arrays to a file using pickle
    with open(outs_path, "wb") as file:
        pickle.dump(outs_f, file)
    logger.info("finished successfully")
