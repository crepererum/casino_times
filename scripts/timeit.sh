#!/usr/bin/env bash

# warm-up & debug
$@

# measure
time for i in {1..20}; do $@ > /dev/null; done
