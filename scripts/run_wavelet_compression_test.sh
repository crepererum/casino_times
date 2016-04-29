#!/usr/bin/env bash

set -euo pipefail

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..

ngram=$1
maxerror=$2

tmpdir=$dir/tmp
datadir=$dir/data
scriptdir=$dir/scripts
binarydir=$dir/cpp/build

ylength=256
dtw_r=10
indexsize=$(expr 1024 \* 1024 \* 1024 \* 4)
inputfile=$datadir/v0-forwavelet.data
mapfile=$datadir/final.map
reffile=$datadir/ref-dists.datax
reportfile=$dir/wavelet_compression_test.report

logfile=$tmpdir/wavelet_compression_test.log
indexfile=$tmpdir/wavelet.index
dumpfile=$tmpdir/dump.data
dtwinfile=$tmpdir/dtwin.data
distfile=$tmpdir/dist.datax

echo "prepare..."
mkdir -p $tmpdir
echo "done"

echo "write intro to report..."
echo "========== BEGIN ==========" >> $reportfile
echo "ngram=$ngram" >> $reportfile
echo "maxerror=$maxerror" >> $reportfile
echo "done"

echo "run wavelet compression..."
LD_PRELOAD=/usr/lib/libtcmalloc.so $binarydir/index_wavelet \
    --binary $inputfile \
    --error $maxerror \
    --index $indexfile \
    --map $mapfile \
    --size $indexsize \
    --ylength $ylength >> $logfile
grep "level=" $logfile >> $reportfile
echo "done"

echo "dump compressed data back into plain file..."
$binarydir/dump_wavelet_index \
    --binary $dumpfile \
    --index $indexfile \
    --map $mapfile \
    --ylength $ylength >> $logfile
echo "done"

echo "prepare data for DTW..."
$scriptdir/transform_data.jl \
    $mapfile \
    $dumpfile \
    $dtwinfile \
    transform_gradient_smooth1 \
    Float64 >> $logfile
echo "done"

echo "calc DTW distances..."
$binarydir/calc_dtw_simple \
    --binary $dtwinfile \
    --map $mapfile \
    --output $distfile \
    --query $ngram \
    --r $dtw_r \
    --ylength $ylength >> $logfile
echo "done"

echo "compare results..."
$scriptdir/compare_dists.jl \
    $mapfile \
    $reffile \
    $distfile | tee --append $logfile $reportfile > /dev/null
echo "done"

echo "write outro to report..."
echo "=========== END ===========" >> $reportfile
echo "done"

echo "clean up..."
rm $logfile $indexfile $dumpfile $dtwinfile $distfile
echo "done"
