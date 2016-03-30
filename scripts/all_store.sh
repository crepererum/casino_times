#!/usr/bin/env bash

set -euo pipefail

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ystart=1753
ylength=256
input_file=$dir/../data/googlebooks-eng-all-1gram-20120701-
map=$dir/../data/final.map
trans=$dir/../data/stems.trans

storage_v0=$dir/../data/v0.data
storage_v1=$dir/../data/v1.data

echo "create storage files"
$dir/../cpp/build/create --map $map --binary0 $storage_v0 --binary1 $storage_v1 --ylength $ylength
echo "done"

for letter in {a..z}; do
    echo
    echo "store data: $letter"
    $dir/../cpp/build/store --file $input_file$letter --binary0 $storage_v0 --binary1 $storage_v1 --map $map --ystart $ystart --ylength $ylength --trans $trans
    echo "done"
done

echo
echo "finished!"
