
import numpy as np

from scipy import sparse

import snowballstemmer

import pickle

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

    def __init__(self):
        self.current_id = 0
        self.vocab = {}
        self.inv_vocab = {}

    def add(self, word):
        if self.vocab.get(word) is None:
            self.vocab[word] = self.current_id
            self.inv_vocab[self.current_id] = word

            self.current_id += 1

    def add_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                for word in line.split():
                        self.add(word)

def build_vocab(infiles, voc_out, inv_voc_out):

    stemmer = MemoizedStemmer()
    vb = VocabularyBuilder()

    for filename in infiles:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                for word in line.split():
                    stemmed = stemmer.stem(word)
                    vb.add(stemmed)

    with open(voc_out, 'wb') as f:
        pickle.dump(vb.vocab, f, protocol=-1)

    with open(inv_voc_out, 'wb') as f:
        pickle.dump(vb.inv_vocab, f, protocol=-1)

def load_vocab(vocab_file, inv_vocab_file):

    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)

    with open(inv_vocab_file, 'rb') as f:
        inv_vocab = pickle.load(f)

    return vocab, inv_vocab

def build_cooc(infiles, vocab, cooc_out):

    rows = []
    cols = []
    data = []

    stemmer = MemoizedStemmer()

    for filename in infiles:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                stemmed = [stemmer.stem(w) for w in line.split()]
                tokens = [vocab[w] for w in stemmed]

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

def tweet_to_seq(words, vocab):
    return [vocab[word] for word in words]

