#!/usr/bin/env bash

set -euo pipefail

cat $1 | grep -E -v "[_.,:;!?/\\\\'\"#()<>=+*{}0-9]" > $2
