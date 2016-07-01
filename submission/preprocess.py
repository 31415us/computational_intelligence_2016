
import numpy as np

from scipy import sparse

import snowballstemmer

import pickle

class StopWordFilter(object):

    def __init__(self, stopword_file):
        self.stopwords = set()
        with open(stopword_file, 'r', encoding='utf-8') as f:
            for line in f:
                for stopword in line.split():
                    self.stopwords.add(stopword)

    def filter(self, word_list):
        return [word for word in word_list if word not in self.stopwords]

class MemoizedStemmer(object):

    def __init__(self):
        self.stemmer = snowballstemmer.stemmer('english')
        self.lookup = {}

    def stem(self, word):
        res = self.lookup.get(word)
        
        if res is None:
            stemmed = self.stemmer.stemWord(word)
            self.lookup[word] = stemmed
            res = stemmed

        return res

class VocabularyBuilder(object):

    def __init__(self, cutoff=10):
        self.cutoff = cutoff
        self.id = 0
        self.vocab = {}
        self.inv_vocab = {}

        self.count = {}

    def add(self, word):
        if self.vocab.get(word) is None:
            if self.count.get(word) is None:
                self.count[word] = 1
            else:
                self.count[word] += 1

            if self.count[word] >= self.cutoff:
                self.vocab[word] = self.id
                self.inv_vocab[self.id] = word
                self.id += 1

    def build(self):
        return self.vocab, self.inv_vocab


def build_vocab(infiles, stopword_file, voc_out, inv_voc_out, cutoff=10):

    stemmer = MemoizedStemmer()
    stopwords = StopWordFilter(stopword_file)
    vb = VocabularyBuilder(cutoff)

    for filename in infiles:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.split()
                filtered = stopwords.filter(words)
                for word in filtered:
                    stemmed = stemmer.stem(word)
                    vb.add(stemmed)

    vocab, inv_vocab = vb.build()

    with open(voc_out, 'wb') as f:
        pickle.dump(vocab, f, protocol=-1)

    with open(inv_voc_out, 'wb') as f:
        pickle.dump(inv_vocab, f, protocol=-1)

def build_cooc(infiles, vocab, cooc_out):

    rows = []
    cols = []
    data = []

    stemmer = MemoizedStemmer()

    for filename in infiles:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                stemmed = [stemmer.stem(w) for w in line.split()]
                tokens = [vocab.get(w) for w in stemmed if vocab.get(w) is not None]

                for t1 in tokens:
                    for t2 in tokens:
                        rows.append(t1)
                        cols.append(t2)
                        data.append(1)

    cooc = sparse.coo_matrix((data, (rows, cols)))
    cooc.sum_duplicates()

    with open(cooc_out, 'wb') as f:
        pickle.dump(cooc, f, protocol=-1)

def load_pickled(infile):
    with open(infile, 'rb') as f:
        result = pickle.load(f)

    return result

