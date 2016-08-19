#!/usr/bin/env sh

folder=img

for f in $folder/*.svg; do
    base=$(basename $f .svg)
    echo converting $base...
    inkscape -D -z --file=$folder/$base.svg --export-pdf=$folder/$base.pdf
done
