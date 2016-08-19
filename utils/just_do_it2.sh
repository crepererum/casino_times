#!/usr/bin/env bash

set -euo pipefail

clevel=$1
ngram=$2

for mw in {0.0000000000001,0.000000000001,0.00000000001,0.0000000001,0.000000001,0.00000001,0.0000001,0.000001}; do
    for r in {0,5,10,15,20}; do
        echo "=================================="
        echo "$clevel,$ngram,$mw,$r"
        ./cpp/build/query_dtw_wavelet --binary data/v0-dtw_smooth1.data --map data/final.map --wavelet cpp/build/jdx.$clevel --ylength 256 --r $r --limit 20 --query $ngram --maxdepth 5 --minweight $mw
    done
done
