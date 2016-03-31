#!/usr/bin/env bash

tail -n+3 | sed -e 's/^|\s*\(\w*\).*/"\1"/' | paste -d"," -s | sed 's/,/, /g'
