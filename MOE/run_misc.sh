#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# . ./path.sh || exit 1;
# . ./cmd.sh || exit 1;

declare -A mappings=( ["decode_dev_decode_ctc_nolm"]="Dev" ["decode_test_mcv_test_decode_lm"]="Test")

for expdir in final_exps/ctc_ce; do

    expdir=exp/$expdir
    echo ""
    echo "$expdir"
    for decodedir in decode_test_mcv_test_decode_lm; do 
        echo "${mappings[$decodedir]}"
        python scripts/xer_decode_scraper.py -r <(cut $expdir/$decodedir/ref.wrd.trn -d'(' -f1) -h <(cut $expdir/$decodedir/hyp.wrd.trn -d'(' -f1)
        # python /espnet/egs/librispeech_100/asr1/scripts/xer_decode_scraper.py -r <(cut $expdir/$decodedir/ref.wrd.trn -d'(' -f1) -h <(cut $expdir/$decodedir/hyp.wrd.trn -d'(' -f1)
        cd $expdir/$decodedir
        python ../../../../scripts/mcv_decode_scraper_accent_specific.py
        cd ../../../
    done;
done;

