#!/usr/bin/env bash

set -euo pipefail

cat $1 | sed -e 's/.* -> //' | sort | uniq > $2
