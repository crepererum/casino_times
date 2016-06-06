#!/usr/bin/env bash

set -euo pipefail

transfile=$1
ngram=$2

exec grep --color=never -e "^$ngram -> " $transfile
