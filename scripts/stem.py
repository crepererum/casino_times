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
    words = ngram_stripped.split(' ')
    words_corrected = words  #fix_spelling(...)
    words_lem = (
        wnl.lemmatize(w, 'v')
        for w in words_corrected
    )
    words_stem = (
        sbs.stem(w)
        for w in words_lem
    )
    ngram_final = ' '.join(words_stem)
    args.fout.write('{} -> {}\n'.format(ngram_stripped, ngram_final))
