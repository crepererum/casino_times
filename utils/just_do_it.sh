#!/usr/bin/env bash

set -euo pipefail

clevel=$1
ngram=$2

for md in {0,1,2,3,4,5,6}; do
    for r in {0,5,10,15,20}; do
        echo "=================================="
        echo "$clevel,$ngram,$md,$r"
        ./cpp/build/query_dtw_wavelet --binary data/v0-dtw_smooth1.data --map data/final.map --wavelet cpp/build/jdx.$clevel --ylength 256 --r $r --limit 20 --query $ngram --maxdepth $md --minweight 0
    done
done
