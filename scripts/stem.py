#!/usr/bin/env python3

import argparse

import nltk.stem

parser = argparse.ArgumentParser()
parser.add_argument('fin', type=argparse.FileType('r'))
parser.add_argument('fout', type=argparse.FileType('w'))
args = parser.parse_args()

wnl = nltk.stem.WordNetLemmatizer()
sbs = nltk.stem.snowball.EnglishStemmer()

for ngram in args.fin:
    ngram_stripped = ngram.strip()
    ngram_lem = wnl.lemmatize(ngram_stripped, 'v')
    ngram_stem = sbs.stem(ngram_lem)
    args.fout.write('{} -> {}\n'.format(ngram_stripped, ngram_stem))
