#!/usr/bin/env sh

cat $1 | grep -E -v '[_.0-9]' > $2
