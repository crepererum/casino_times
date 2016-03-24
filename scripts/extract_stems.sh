#!/usr/bin/env sh

cat $1 | sed -e 's/[^ ]* -> //' | sort | uniq > $2
