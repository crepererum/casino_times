#!/usr/bin/env python3

import argparse

import enchant
import nltk.stem

parser = argparse.ArgumentParser()
parser.add_argument('fin', type=argparse.FileType('r'))
parser.add_argument('fout', type=argparse.FileType('w'))
args = parser.parse_args()

d = enchant.Dict('en')
wnl = nltk.stem.WordNetLemmatizer()
sbs = nltk.stem.snowball.EnglishStemmer()


def fix_spelling(word):
    if d.check(word):
        return word
    else:
        for s in d.suggest(word):
            if len(s) == len(word) and s.find(' ') == -1 and s.find('-') == -1:
                return s
        return word


for ngram in args.fin:
    ngram_stripped = ngram.strip()
    ngram_corrected = ngram_stripped  #fix_spelling(ngram_stripped)
    ngram_lem = wnl.lemmatize(ngram_corrected, 'v')
    ngram_stem = sbs.stem(ngram_lem)
    args.fout.write('{} -> {}\n'.format(ngram_stripped, ngram_stem))
