#!/usr/bin/env bash

set -euo pipefail

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

letter=$1
ystart=1753
ylength=256
prune_threshold=1000
input_file=$dir/../data/googlebooks-eng-all-1gram-20120701-$letter
map_raw=$dir/../data/$letter-raw.map
map_filtered=$dir/../data/$letter-filtered.map
map_normal=$dir/../data/$letter-normalized.map
map_stems=$dir/../data/$letter-stems.map
map_final=$dir/../data/$letter-final.map
trans_stem=$dir/../data/$letter-stems.trans

storage_pre_v0=$dir/../data/${letter}-v0.data
storage_pre_v1=$dir/../data/${letter}-v1.data

venv_python=~/venv3.5/bin/python

echo "Handle letter $letter..."

echo
echo "1. scan google data"
$dir/../cpp/build/scan --file $input_file --map $map_raw
echo "done"

echo
echo "2. filter"
$dir/../scripts/filter.sh $map_raw $map_filtered
echo "done"

echo
echo "3. normalize"
$dir/../cpp/build/normalize --mapin $map_filtered --mapout $map_normal
echo "done"

echo
echo "4. stemming"
$venv_python $dir/../scripts/stem.py $map_normal $trans_stem
echo "done"

echo
echo "5. extract map from stem results"
$dir/../scripts/extract_stems.sh $trans_stem $map_stems
echo "done"

echo
echo "6. create storage files"
$dir/../cpp/build/create --map $map_stems --binary0 $storage_pre_v0 --binary1 $storage_pre_v1 --ylength $ylength
echo "done"

echo
echo "7. store data"
$dir/../cpp/build/store --file $input_file --binary0 $storage_pre_v0 --binary1 $storage_pre_v1 --map $map_stems --ystart $ystart --ylength $ylength --trans $trans_stem
echo "done"

echo
echo "8. pruning"
$dir/../scripts/prune_support.jl $storage_pre_v0 $map_stems $map_final $prune_threshold
echo "done"

echo
echo "9. report"
n_raw=`wc -l $map_raw | cut -f1 -d' '`
n_filtered=`wc -l $map_filtered | cut -f1 -d' '`
n_normal=`wc -l $map_normal | cut -f1 -d' '`
n_stems=`wc -l $map_stems | cut -f1 -d' '`
n_final=`wc -l $map_final | cut -f1 -d' '`
echo "| letter | raw | filtered | normalized | stems | final |"
echo "| $letter | $n_raw | $n_filtered | $n_normal | $n_stems | $n_final |"
